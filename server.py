import logging
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
import ttl_cache

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
        for i, validator_hk in enumerate(self.positions.keys()):
            i = i % 2
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
        hash_key = self.hash_texts_and_hk(texts, validator_hk)
        if hash_key in self.CACHE:
            return self.CACHE[hash_key]
        
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
        self.CACHE[hash_key] = final_result
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

@app.post("/texts/distances")
def text_distances(request: TextRequest):
    logging.info(f"Request validator: {request.validator}")
    try:
        sim = e_manager.get_distances(request.texts, request.validator)
        logging.info(f"distances: {sim}")
        return {"distances": sim}
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"distances": None}
    
if __name__=='__main__':
    import argparse
    import threading
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    
    parser = argparse.ArgumentParser(description='Embedding service')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    args = parser.parse_args()
    
    e_manager = EmbeddingManager()
    threading.Thread(target=refresh_cache, args=(e_manager,), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=log_config)
