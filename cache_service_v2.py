import redis
from typing import List
from starlette.concurrency import run_in_threadpool
import json
import logging
import os
import time
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import uvicorn
import logging
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import logging
import torch
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

executor = ThreadPoolExecutor(max_workers=20)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def get_model(device="cuda:0"):
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L12-v2',
        trust_remote_code=True,
        device=device
    )
    return model


class EmbeddingManagerV2:
    
    def get_device(self, part: int, gpus: int):
        return f"cuda:{part % gpus}"
        
    def __init__(self, embeds_dir, parts) -> None:
        self.batch_size = 5000
        self.FILES_PER_PART = 30
        self.cos_layer = torch.nn.CosineSimilarity(dim=2)
        # euclidean distance layer
        self.euclidean_layer = torch.nn.PairwiseDistance(p=2)
        
        self.embeddings_dir = embeds_dir
        self.parts = [int(p) for p in parts.split(",")]
        self.load_embeddings()

        self.model = get_model("cpu")
        self.MAX_FILE = 16
        self.CACHE = {}

    def load_embeddings(self):
        
        self.embeddings_by_devices = {
            "cuda:0": None,
            "cuda:1": None
        }
        for cuda in range(torch.cuda.device_count()):
            self.embeddings_by_devices[f"cuda:{cuda}"] = None

        import os
        for i, part in enumerate(self.parts):
            device = self.get_device(i, torch.cuda.device_count())
            if self.embeddings_by_devices[device] is None:
                self.embeddings_by_devices[device] = np.load(f"{self.embeddings_dir}/{part}.npy")
            else:
                self.embeddings_by_devices[device] = np.concatenate([self.embeddings_by_devices[device], np.load(f"{self.embeddings_dir}/{part}.npy")], axis=0)
        print(f"Loaded embeddings for {len(self.embeddings_by_devices)} devices")
        for device in self.embeddings_by_devices.keys():
            print(f"Load embeddings for device: {device}")
            self.embeddings_by_devices[device] = torch.from_numpy(self.embeddings_by_devices[device]).to(device)
            
    def preprocess(self, texts: str):
        sentences = []
        indices = []
        for i, text in enumerate(texts):
            split_text = sent_tokenize(text)
            sentences.extend(split_text)
            indices += [i]*len(split_text)
        idx = {}
        for i in range(len(indices)):
            if indices[i] not in idx:
                idx[indices[i]] = []
            idx[indices[i]].append(i)
        return sentences, idx

    def hash_texts_and_hk(self, texts: list[str], validator_hk: str):
        import hashlib
        import json
        data = {
            "texts": texts,
            "validator_hk": validator_hk
        }
        data = json.dumps(data)
        return hashlib.md5(data.encode()).hexdigest()

    def get_distances(self, texts: str, validator_hk: str):
        sentences, indices = self.preprocess(texts)
        target_device = self.device_by_validator[validator_hk]
        embs = self.model.encode(sentences)
        embs = torch.from_numpy(embs).to(target_device)
        result = []
        target_embeddings = self.embeddings_by_validator[validator_hk]
        for i in range(0, len(target_embeddings), self.batch_size):
            batch = target_embeddings[i:i+self.batch_size]
            sim = torch.cdist(embs, batch).min(dim=1).values
            result.append(sim.unsqueeze(0))

        result = torch.cat(result, dim=0).min(dim=0).values.tolist()

        final_result = []
        for i in range(len(texts)):
            ids = indices[i]
            final_result.append(sum([result[j] for j in ids]) / len(ids))
        return final_result

    def distances(self, text_embeddings: torch.Tensor, embeddings: torch.Tensor):
        result = []
        for i in range(0, len(embeddings), self.batch_size):
            batch = embeddings[i:i+self.batch_size]
            sim = torch.cdist(text_embeddings, batch).min(dim=1).values
            result.append(sim.unsqueeze(0).to("cpu"))
        return result
        # return torch.cat(result, dim=0).min(dim=0).values.tolist()
    
    def get_distances_v2(self, texts: str):
        
        sentences, indices = self.preprocess(texts)
        embs = self.model.encode(sentences)
        result = []
        futures = []
        for device, embeddings in self.embeddings_by_devices.items():
            print(f"text embeddings for {device}")
            target_embs = torch.from_numpy(embs).to(device)
            future = executor.submit(self.distances, target_embs, embeddings)
            futures.append(future)
        
        for future in futures:
            result.extend(future.result())
        result = torch.cat(result, dim=0).min(dim=0).values.tolist()

        final_result = []
        for i in range(len(texts)):
            ids = indices[i]
            final_result.append(sum([result[j] for j in ids]) / len(ids))
        return final_result
    
    
def refresh_cache(e_manager):
    while True:
        print(f"Refreshing cache...")
        e_manager.refresh_cache()
        print(f"Cache refreshed and sleeping for 1 hour")
        time.sleep(60*60)


app = FastAPI()


@app.get("/")
def healthcheck():
    return {"status": "ok"}


class TextRequest(BaseModel):
    texts: List[str]
    validator:  str


CACHE = redis.Redis(host='localhost', port=30379, db=0)


def get_response_from_cache(hash_key: str, timeout: int = 10):
    start = time.time()
    while True:
        value = CACHE.get(hash_key).decode()
        if value != "":
            logger.info(f"CACHE HIT for {hash_key}")
            return json.loads(value)

        if time.time() - start > timeout:
            logger.info(f"Timeout for {hash_key}")
            return None


@app.post("/texts/distances")
async def text_distances(text_req: TextRequest):
    # if len(text_req.texts) == 300:
    #     return {"distances": None}
    # hash_key = e_manager.hash_texts_and_hk(text_req.texts, text_req.validator)

    # exists = CACHE.exists(hash_key)
    # if exists:
    #     print(f"GET from cache: {hash_key}, validator: {text_req.validator}")
    #     distances = await run_in_threadpool(get_response_from_cache, hash_key)
    #     return {"distances": distances}

    # logger.info(f"Request validator: {text_req.validator}")
    # CACHE.set(hash_key, "")
    try:
        sim = await run_in_threadpool(e_manager.get_distances_v2, text_req.texts)
        logging.info(f"distances: {sim}")
        # CACHE.set(hash_key, json.dumps(sim))
        return {"distances": sim}

    except Exception as e:
        logging.error(f"Error: {e}")
        return {"distances": None}

if __name__ == '__main__':
    import argparse
    import threading
    parser = argparse.ArgumentParser(description='Embedding service')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--embeds_dir', type=str, default="embeddings/0", help='Embeddings directory')
    parser.add_argument("--parts", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", help="Parts")
    args = parser.parse_args()
    
    e_manager = EmbeddingManagerV2(args.embeds_dir, args.parts)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
