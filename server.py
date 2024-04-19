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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def get_model(device="cuda:0"):
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L12-v2',
        trust_remote_code=True,
        device=device
    )
    return model

class EmbeddingManager:
    def __init__(self, keys_path="validator_positions.json") -> None:
        self.batch_size = 5000
        self.FILES_PER_PART = 30
        self.cos_layer = torch.nn.CosineSimilarity(dim=2)
        # euclidean distance layer
        self.euclidean_layer = torch.nn.PairwiseDistance(p=2)
        self.embeddings_by_validator = {}
        self.load_positions()
        self.load_embeddings()

        self.model = get_model("cpu")
        
        self.CACHE = {}
        
        
    def load_positions(self):
        import json
        with open("validator_positions.json", "r") as f:
            self.positions = json.load(f)
        
        self.device_by_validator = {}
        VALIDATORS = list(self.positions.keys())
        # get list cuda devices
        cuda_count = torch.cuda.device_count()
        print(f"Found {cuda_count} cuda devices.")
        for i, validator_hk in enumerate(self.positions.keys()):
            i = i % cuda_count
            self.device_by_validator[validator_hk] = f"cuda:{i}"
            
    def load_embeddings(self):
        import os
        for i, validator_hk in enumerate(self.positions.keys()):
            device = self.device_by_validator[validator_hk]
            validator_position = self.positions[validator_hk]
            fpath = f"embeddings/{validator_position}.npy"
            if os.path.exists(fpath):
                embedding = np.load(fpath)
                self.embeddings_by_validator[validator_hk] = torch.from_numpy(embedding).to(device)
                print(f"Loaded embeddings for {validator_hk} with shape: {self.embeddings_by_validator[validator_hk].shape}")
            else:
                print(f"Embeddings for {validator_hk} not found.")
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
    
    def hash_texts_and_hk(self, texts: str, validator_hk: str):
        import hashlib
        texts = "".join(texts)
        return hashlib.md5(f"{texts}{validator_hk}".encode()).hexdigest()
    
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
    
    def refresh_cache(self):
        self.CACHE = {}

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

from typing import List
class TextRequest(BaseModel):
    texts: List[str]
    validator:  str

import redis
CACHE = redis.Redis(host='localhost', port=6379, db=0)

@app.post("/texts/distances")
def text_distances(text_req: TextRequest):
    hash_key = e_manager.hash_texts_and_hk(text_req.texts, text_req.validator)
    cached_response =  CACHE.get(hash_key)
    if cached_response:
        logger.info(f"CACHE HIT: {hash_key}")
        distances = json.loads(cached_response)
        return {"distances": distances}
    
    logger.info(f"Request validator: {text_req.validator}")
    try:
        sim = e_manager.get_distances(text_req.texts, text_req.validator)
        logging.info(f"distances: {sim}")
        CACHE.set(hash_key, json.dumps(sim))
        return {"distances": sim}
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"distances": None}
    
if __name__=='__main__':
    import argparse
    import threading
    parser = argparse.ArgumentParser(description='Embedding service')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    args = parser.parse_args()
    
    e_manager = EmbeddingManager()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
