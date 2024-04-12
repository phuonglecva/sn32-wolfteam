# write code to download content from the given URL and save to download folder
import os

def download_file(part):
    file_path = f"download/c4-train.{part}-of-01024.json.gz"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        print(f"File '{file_path}' already exists.")
        return
    # write code to download content from the given URL and save to download folder
    import requests
    url = f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{part}-of-01024.json.gz"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"File '{file_path}' downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download file from given URL')
    parser.add_argument('part', type=int, help='Part number')
    args = parser.parse_args()
    
    # build part as 5 digit, zero padded string
    part = str(args.part).zfill(5)
    download_file(part)