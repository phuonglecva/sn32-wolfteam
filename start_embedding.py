import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer


def embeddings(fpath: str, part: int, model: SentenceTransformer, pool, batch_size: int = 128):
    """
    Args:
        fpath (str): path to texts file
        part (int): part number
        model (SentenceTransformer): SentenceTransformer model
        pool (_type_): cuda devices
        batch_size (int, optional): batch size. Defaults to 128.
    """
    start = time.time()
    with open(f"output/{fpath}.json", "r", encoding="utf8") as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts from '{fpath}'.")    
    print(f"Embedding {len(texts)} texts.")
    embeddings = model.encode_multi_process(texts, pool, batch_size=batch_size)
    print(f"Embeddings shape: {embeddings.shape}, time taken: {time.time() - start} seconds.")
    os.makedirs("embeddings/part", exist_ok=True)

    with open(f"embeddings/part/{fpath}.npy", "wb") as f:
        np.save(f, embeddings)
    print(f"Embeddings for '{fpath}' saved successfully. Time taken: {time.time() - start} seconds.")
    

def get_model():
    model = SentenceTransformer(
        'jinaai/jina-embeddings-v2-small-en',
        trust_remote_code=True
    )
    model.max_seq_length = 1024
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Embedding service')
    parser.add_argument('--part', type=int, help='Part number')
    parser.add_argument('--devices', type=str, default="0")
    
    args = parser.parse_args()
    devices = args.devices.split(',')   
    if len(devices) == 0:
        print("No devices specified.")
        exit(0)
    devices = [f"cuda:{d}" for d in devices]
    
    model = get_model()
    pool = model.start_multi_process_pool(target_devices=devices)
    part = str(args.part).zfill(5)
    file_path = f'output/{part}.json'
    embeddings(file_path, args.part, model, pool)