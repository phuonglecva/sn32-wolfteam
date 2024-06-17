from validation import infer_distance
import json
import os

if __name__ == '__main__':

    input_dir = '/root/sample_data/5CXC2quDN5nUTqHMkpP5YRp2atYYicvtUghAYLj15gaUFwe5'
    files = os.listdir(input_dir)

    for file in files:
        file_path = os.path.join(input_dir, file)
        print(f'file_path = {file_path}')
        with open(file_path, 'r') as f:
            data = json.load(f)
            texts = data['texts']
            validator_hotkey = data['validator_hotkey']
            infer_distance(texts, validator_hotkey)

