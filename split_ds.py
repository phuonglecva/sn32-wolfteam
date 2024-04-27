import argparse

parser = argparse.ArgumentParser(description='Split datasets for embedding')
parser.add_argument('--path', type=str, help='Path to the dataset')

args = parser.parse_args()

import json
from nltk import sent_tokenize
MAX_ROWS = 7_500_000
import os

texts = []
current_part = 1
with open(args.path, "r") as f:
    for line in f:
        row = json.loads(line)
        text = row['text']
        sentences = sent_tokenize(text) 
        texts.extend(sentences)
        
        if len(texts) >= MAX_ROWS:
            save_path = f"output/{args.path}/{current_part}.json"
            
            print(f"Saving {len(texts)} rows to {save_path}")
            os.makedirs("output", exist_ok=True)
            with open(save_path, 'w', encoding="utf8") as f:
                json.dump(texts, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(texts)} rows to {save_path}")
            texts = []
            current_part += 1
            

