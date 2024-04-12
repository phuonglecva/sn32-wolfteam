import logging
import time
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import os
import logging
from argparse import ArgumentParser 
parser = ArgumentParser()
parser.add_argument("--file_path", type=str, default="")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--port", type=int, default=51003)
parser.add_argument("--part", type=int, default=0)
args = parser.parse_args()
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
class EmbeddingManager:
    def __init__(self, device="cuda", file_path="", part=0) -> None:
        self.batch_size = 5000
        self.FILES_PER_PART = 30
        self.file_path = file_path
        self.THRESHOLD = 0.99
        self.cos_layer = torch.nn.CosineSimilarity(dim=2)
        self.embeddings = []
        self.part = part
        self.device = device
        self.load_embeddings()

    def load_embeddings(self):
        import os
        fpaths = [f"output/{f}" for f in os.listdir("output") if f.endswith(".npy") and "train" in f]
        fpaths = fpaths[self.FILES_PER_PART * self.part: self.FILES_PER_PART * (self.part + 1)]
        print(f"Loading embeddings from {fpaths}")
        for i, fpath in enumerate(fpaths):
            print(f"Loaded {i}/{len(fpaths)}")
            with open(fpath, "rb") as f:
                if len(self.embeddings) == 0:
                    self.embeddings = np.load(f)
                else:
                    self.embeddings = np.vstack([self.embeddings, np.load(f)])
        self.embeddings = torch.from_numpy(self.embeddings).to(self.device)
        self.fpaths = fpaths
        self.part_ids = [int(f.split(".")[1].split("-")[0]) for f in fpaths]
        print(f"Loaded embeddings with shape: {self.embeddings.shape}")
    
    def get_similar_with_emb(self, embedding):
        start = time.time() 
        embedding = torch.Tensor(embedding).to(self.device)  
        max_score = 0.0
        for i in range(0, len(self.embeddings), self.batch_size):
            batch = self.embeddings[i:i+self.batch_size].to(self.device)
            sim = self.cos_layer(embedding, batch)
            if sim.max() > self.THRESHOLD:
                return sim.max().item()
            max_score = max(max_score, sim.max().item())
        print(f"Time: {time.time() - start}")   
        return max_score            
    
    def get_similarities(self, embs):
        embs = embs.unsqueeze(1)
        result = []
        for i in range(0, len(self.embeddings), self.batch_size):
            batch = self.embeddings[i:i+self.batch_size].unsqueeze(0).to(self.device)
            sim = self.cos_layer(embs, batch)
            result.append(sim.max(dim=1).values.unsqueeze(0))
        
        result = torch.cat(result, dim=0)
        return result.max(dim=0).values.tolist()
    
app = FastAPI()
e_manager = EmbeddingManager(file_path=args.file_path, part=args.part, device=args.device)
@app.get("/")
def healthcheck():
    return {"status": "ok"}

class Req(BaseModel):
    text: str

from typing import List

class EmbeddingReq(BaseModel):
    embedding: list[float]

class EmbeddingsReq(BaseModel):
    embeddings: List[list[float]]
    
@app.post("/similar")
def similar(request: EmbeddingReq):
    # logging.info(f"Request: {request}")
    try:
        sim = e_manager.get_similar_with_emb(request.embedding)
        # logging.info(f"Similarity: {sim}")
        return {"similarity": sim}
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"similarity": None}

@app.get("/part/ids")
def get_part_ids():
    return {
        "ids": e_manager.part_ids
    }
    
@app.post("/similarities")
def similarities(request: EmbeddingsReq):
    # logging.info(f"Request: {request}")
    try:
        embs = torch.Tensor(request.embeddings).to(args.device)
        sim = e_manager.get_similarities(embs)
        logging.info(f"Similarities: {sim}")
        return {"similarities": sim}
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"similarities": None}
if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config="log_config.yaml")
