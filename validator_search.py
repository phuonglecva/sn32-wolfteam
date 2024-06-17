from validation import infer_distance
import json
import os

if __name__ == '__main__':

    input_dir = '/root/sample_data/'
    # files = os.listdir(input_dir)
    directories = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            file_path = os.path.join(directory, file)
            print(f'file_path = {file_path}')
            with open(file_path, 'r') as f:
                data = json.load(f)
                texts = data['texts']
                if len(texts) == 300:
                    validator_hotkey = data['validator_hotkey']
                    infer_distance(texts, validator_hotkey)

