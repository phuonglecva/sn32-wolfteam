import asyncio
from typing import List
import fastapi
import json
import aiohttp


class ValidatorService:
    def __init__(self) -> None:
        self.load_hosts()
        
        self.validator_hosts = {}
        
    def load_hosts(self):
        with open("validator_hosts.json", "r") as f:
            self.validator_hosts = json.load(f)

    
    async def get_distance(self, texts: List[str], validator: str):
        # 1 validator can map to multiple hosts
        hosts = self.validator_hosts.get(validator)
        if hosts is None:
            print(f"Validator {validator} not found")
            return None
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for host in hosts:
                url = f"http://{host}/texts/distances"
                data = {
                    "texts": texts,
                    "validator": validator
                }
                tasks.append(session.post(url, json=data))
            
            responses = await asyncio.gather(*tasks)
            results = []
            for response in responses:
                results.append(await response.json())
            return results
    
app = fastapi.FastAPI()

validator_service = ValidatorService()

@app.get("/")
def read_root():
    return {"Hello": "World"}

from pydantic import BaseModel

class TextRequest(BaseModel):
    texts: List[str]
    validator: str

@app.post("/texts/distances")
async def read_item(text_req: TextRequest):
    distance = await validator_service.get_distance(text_req.texts, text_req.validator)
    return {"distances": distance}