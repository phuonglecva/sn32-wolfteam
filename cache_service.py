import json
import sys
import redis
from fastapi import FastAPI
import hashlib
from pydantic import BaseModel
import logging
import logging.config
import yaml
import argparse
app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

LOGGER = logging.getLogger(__name__)
DISTANCE_SERVICE_URL="http://localhost:8899"

CACHE = redis.Redis(host='localhost', port=6379, db=0)

class TextRequest(BaseModel):
    texts: list[str]
    validator: str    

import aiohttp
def build_hash(texts: list[str], validator: str) -> str:
    texts = "".join(texts)
    return hashlib.md5(f"{texts}{validator}".encode()).hexdigest()

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/texts/distances")
async def text_distances(request: TextRequest):
    """
    write code using redis lock to perform only one time with same request body
    """
    hash_key = build_hash(request.texts, request.validator)   
    LOGGER.info(f"hash_key: {hash_key}")
    cached_response =  CACHE.get(hash_key)
    if cached_response:
        LOGGER.info(f"CACHE HIT: {hash_key}")
        distances = json.loads(cached_response)
        return {"distances": distances}
    lock = CACHE.lock(f"Lock:{hash_key}", blocking=False)
    with lock:
        LOGGER.info(f"Request validator: {request.validator}")
        cached_response = CACHE.get(hash_key)
        if cached_response:
            distances   = json.loads(cached_response)
            return {"distances": distances}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{DISTANCE_SERVICE_URL}/texts/distances", json={
                "texts": request.texts,
                "validator": request.validator
            }) as response:
                response = await response.json()
                LOGGER.info(f"validator: {request.validator}distances: {response['distances']}")
                CACHE.set(hash_key, json.dumps(response["distances"]))
                return response
            
if __name__ == "__main__":
    import uvicorn 
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()  
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    