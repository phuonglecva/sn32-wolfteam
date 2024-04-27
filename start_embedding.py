import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer


def embeddings(
    fpath: str, part: int, model: SentenceTransformer, 
    pool, output_file: str, 
    batch_size: int = 384):
    """
    Args:
        fpath (str): path to texts file
        part (int): part number
        model (SentenceTransformer): SentenceTransformer model
        pool (_type_): cuda devices
        batch_size (int, optional): batch size. Defaults to 128.
    """
    start = time.time()
    with open(f"{fpath}", "r", encoding="utf8") as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts from '{fpath}'.")    
    print(f"Embedding {len(texts)} texts.")
    embeddings = model.encode_multi_process(texts, pool, batch_size=batch_size)
    print(f"Embeddings shape: {embeddings.shape}, time taken: {time.time() - start} seconds.")
    os.makedirs("embeddings/part", exist_ok=True)

    with open(output_file, "wb") as f:
        np.save(f, embeddings)
    print(f"Embeddings for '{fpath}' saved successfully. Time taken: {time.time() - start} seconds.")
    

def get_model():
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L12-v2',
        trust_remote_code=True
    )
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Embedding service')
    parser.add_argument('--devices', type=str, default="0")
    parser.add_argument("--input_file", type=str, default="data/0.json")
    parser.add_argument("--output_file", type=str, default="output/0/1.npy")
    args = parser.parse_args()
    devices = args.devices.split(',')   
    if len(devices) == 0:
        print("No devices specified.")
        exit(0)
    devices = [f"cuda:{d}" for d in devices]
    
    model = get_model()
    pool = model.start_multi_process_pool(target_devices=devices)
    part = str(args.part).zfill(5)
    embeddings(args.input_file, args.part, model, pool, args.output_file)