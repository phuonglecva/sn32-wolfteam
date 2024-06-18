import random
import shutil

from validation import infer_distance
import json
import os


def search_validator():
    input_dir = '/root/sample_data/5Fq5v71D4LX8Db1xsmRSy6udQThcZ8sFDqxQFwnUZ1BuqY5A'
    # files = os.listdir(input_dir)
    directories = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if
                   os.path.isdir(os.path.join(input_dir, d))]
    for directory in directories:
        files = os.listdir(directory)
        random.shuffle(files)
        count = 0
        for file in files:
            if count == 5:
                break
            file_path = os.path.join(directory, file)
            print(f'file_path = {file_path}')
            with open(file_path, 'r') as f:
                data = json.load(f)
                texts = data['texts']
                if len(texts) == 300:
                    validator_hotkey = data['validator_hotkey']
                    infer_distance(texts, validator_hotkey)
                    count += 1


def move_file():
    directory = '/root/sample_data/'
    files = [file for file in os.listdir(directory) if not os.path.isdir(os.path.join(directory, file))]
    for file in files:
        file_path = os.path.join(directory, file)
        dir_name = file.split('_')[0]
        target_dir = os.path.join(directory, dir_name)
        destination_file = os.path.join(target_dir, file)
        if os.path.exists(destination_file):
            os.remove(destination_file)

        shutil.move(file_path, target_dir)


if __name__ == '__main__':
    search_validator()
