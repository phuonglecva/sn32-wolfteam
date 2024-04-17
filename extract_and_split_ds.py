def read_data(gzip_path): 
    import gzip
    import json
    with open(gzip_path, 'rb') as f:
        with gzip.open(f, 'rt') as file:
            for line in file:
                data = json.loads(line)
                yield data['text']
                
if __name__=='__main__':
    import argparse
    from nltk.tokenize import sent_tokenize
    parser = argparse.ArgumentParser(description='Extract and Split datasets for embedding')
    parser.add_argument('--part', type=int, help='Part number')
    args = parser.parse_args()
    
    # build part as 5 digit, zero padded string
    part = str(args.part).zfill(5)
    
    gzip_path = f"download/c4-train.{part}-of-01024.json.gz"
    result = []
    for text in read_data(gzip_path):
        sentences = sent_tokenize(text)
        result.extend(sentences)
    print(f"Total sentences: {len(result)}")
    # write to file
    import os
    os.makedirs("output", exist_ok=True)
    with open(f"output/{part}.json", 'w', encoding="utf8") as f:
        import json
        json.dump(result, f, indent=2, ensure_ascii=False)