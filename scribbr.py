import json
import time

import requests
from typing import List

from validation import print_accuracy_distance_test

SAMPLE_DATA_DIR = '/root/sample_sent_data_pile22'
# API = "http://130.250.178.211:63443/predict"
# API = "http://130.250.178.211:54337/predict"
# API = "http://90.55.27.227:40175/predict"
API = 'http://8.12.5.23:34370/predict'


def detect(text):
    url = 'https://www.scribbr.com/ai-detector.php'
    apikey = '67cf2e217a04dc0352ce93f266adceab32aa2a36'
    params = {
        'url': url,
        'apikey': apikey,
        'custom_headers': 'true',
    }
    headers = {
        'accept': '*/*',
        'accept-language': 'vi-VN,vi;q=0.9',
        'cache-control': 'max-age=0',
        'content-type': 'application/json',
        'cookie': 'amp_6ae9ad=5O85otuC6ydNuMmc31n5KL...1hrso195k.1hrso49um.4.1.5; cf_clearance=D4SYPoIjKcNIpMtP62h9jdqu8SqjJKLel5glHgrxSK8-1713546009-1.0.1.1-fe7ErZ7pNygHXLRFfB74ygiZWQQ1.e5aB96yrDF5ArRkt_nNqGoEVrR8dXqk3QU.r0RmI.z9vuBd_IVtru5cRw',
        'dnt': '1',
        'origin': 'https://www.scribbr.com',
        'referer': 'https://www.scribbr.com/ai-detector/',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }
    data = json.dumps({
        'lang': 'en',
        "text": text
    })
    response = requests.post('https://api.zenrows.com/v1/', params=params, headers=headers, data=data, timeout=4)
    score = response.json()["data"]["text_score"]
    print(f'score = {score}')
    return score > 0.7


def call_api(texts: List[str]):
    import concurrent.futures
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        result = list(executor.map(is_ai, texts))
        return result


def read_file(file_path):
    try:
        # Open the file and read the content
        with open(file_path, 'r') as file:
            data = json.load(file)  # Convert JSON content to a dictionary
            return data
    except Exception as e:
        print(f"Error: The file {file_path} was {e}.")
    return None


def run_test():
    from pathlib import Path
    directory = Path(SAMPLE_DATA_DIR)
    file_names = [file.name for file in directory.iterdir() if file.is_file()]
    total_text = 0
    total_pred_correct = 0
    for f_name in file_names:
        start_time = time.time_ns()
        f_path = SAMPLE_DATA_DIR + '/' + f_name
        data = read_file(f_path)
        if data is None:
            continue

        first = [False] * 150
        second = [True] * 150
        result = []
        result.extend(first)
        result.extend(second)

        auged_texts = data['auged_texts']
        pred_result = []
        for i in range(1):
            texts = auged_texts[300 * i:300*i + 300]
            print(f"type of texts: {type(texts)}")
            pred = call_api(texts)
            pred_result.extend(pred)

        # if len(result) != len(pred_result):
        #     print(f"len(result) = {len(result)}, len(pred_result) = {len(pred_result)}")
        #     continue
        # comp = [result[i] == pred_result[i] for i in range(len(result))]
        # total_text += len(pred_result)
        # total_pred_correct += comp.count(True)
        end_time = time.time_ns()
        time_processing = (end_time - start_time) // 1000_000
        print(f"time processing: {time_processing} ms")
        print(f"pred_result: {pred_result}")
        # print(
        #     f'file {f_name}, pred wrong: {comp.count(False)}, pred correct: {comp.count(True)}, accuracy: {comp.count(True) / len(pred_result)}, average: {total_pred_correct / total_text}, time_processing = {time_processing}ms')
        prediction = [None] * len(pred_result)
        for i in range(len(pred_result)):
            if pred_result[i] > 0.75:
                prediction[i] = True

        print(f"prediction: {prediction}")
        print_accuracy_distance_test(prediction)

def is_ai(text):
    try:
        url = 'https://www.scribbr.com/ai-detector.php'
        apikey = '67cf2e217a04dc0352ce93f266adceab32aa2a36'
        params = {
            'url': url,
            'apikey': apikey,
            'custom_headers': 'true',
        }
        headers = {
            'accept': '*/*',
            'accept-language': 'vi-VN,vi;q=0.9',
            'cache-control': 'max-age=0',
            'content-type': 'application/json',
            'cookie': 'amp_6ae9ad=5O85otuC6ydNuMmc31n5KL...1hrso195k.1hrso49um.4.1.5; cf_clearance=D4SYPoIjKcNIpMtP62h9jdqu8SqjJKLel5glHgrxSK8-1713546009-1.0.1.1-fe7ErZ7pNygHXLRFfB74ygiZWQQ1.e5aB96yrDF5ArRkt_nNqGoEVrR8dXqk3QU.r0RmI.z9vuBd_IVtru5cRw',
            'dnt': '1',
            'origin': 'https://www.scribbr.com',
            'referer': 'https://www.scribbr.com/ai-detector/',
            'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
        }
        data = json.dumps({
            'lang': 'en',
            "text": text
        })
        response = requests.post('https://api.zenrows.com/v1/', params=params, headers=headers, data=data, timeout=3)
        score = response.json()["data"]["text_score"]
        # print(f"score = {score}")
        # return score >= 0.7
        return score
    except Exception as e:
        print(f'generated an exception: {e}')
        return 0


if __name__ == "__main__":
    run_test()
    # text = "xin and edema toxin-mediated activity in vitro and lethality in vivo and were non-toxic to sensitive cell lines when combined with PA. While OA combined with EFH351A was non-lethal in mice, PA combined with LFE687A was of reduced virulence. Full oprotection of mice against a lethal toxin challenge required injection of mce with PA combined with both LFE687A and EFH351A. The potential use of these full-length, biologically inactive mutant proteins combined with PA as prophylactics or therapeutics is discussed."
    # result = call_api([text])
    # print(result)
