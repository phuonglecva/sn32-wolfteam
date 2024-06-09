# Description: This file contains the code for the validation of the model
import time

from nltk.tokenize import sent_tokenize


def infer_model(texts):
    print(f'start infer_model')
    time_start = time.time_ns()
    import requests
    url = "http://8.12.5.23:34370/predict"
    payload = {
        "list_text": texts
    }
    response = requests.request("POST", url, json=payload, timeout=120)
    result = response.json()["result"]
    print(f'model results: {result}')
    time_end = time.time_ns()
    print(f'time infer model: {(time_end - time_start) // 1000_000}')
    return result


def infer_distance(texts):
    try:
        print(f'start call infer_distance')
        time_start = time.time_ns()
        import requests

        url = "http://174.92.219.240:52173/predict"

        new_texts = []
        index_for_text = {}
        start_i = 0
        for i, text in enumerate(texts):
            sentences = sent_tokenize(text)
            sentences = sentences[-3:]
            for _ in sentences:
                if i not in index_for_text:
                    index_for_text[i] = []
                index_for_text[i].append(start_i)
                # print(index_for_text)
                start_i += 1
            new_texts.extend(sentences)
        # write text to file data.json

        import json
        with open('data.json', 'w') as f:
            json.dump(new_texts, f, indent=2)
        # print(f"Length of new texts: {len(new_texts)}")
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "list_text": new_texts
            })
            response = requests.post(url, data=payload, headers=headers, timeout=120)
        except Exception as e:
            print(f"Error: {e}")
            return [False] * len(texts)

        # print(f"Response: {response.json()}")
        result = response.json()["result"]
        final_result = []
        for i, text in enumerate(texts):
            list_index = index_for_text[i]
            list_result = [result[j] for j in list_index]
            final_result.append(min(list_result))
        distance_result = [score < 0.001 for score in final_result]
        print(f'distance result: {distance_result}')
        time_end = time.time_ns()
        print(f'time processing distance: {(time_end - time_start) // 1000_000} ms')

        return distance_result

    except Exception as e:
        print(f"Error full: {e}")
        return [False] * len(texts)


def infer_with_distance(texts: list[str]):
    distances = infer_distance(texts)
    preds = infer_model(texts)
    result = []
    for i in range(len(distances)):
        if distances[i]:
            result.append(False)
        else:
            result.append(preds[i])
    return result


if __name__ == '__main__':
    import json
    import numpy as np
    import os
    from reward import get_rewards

    labels = [False] * 150 + [True] * 150
    input_dir = '/root/.phuong/sample_data'
    files = os.listdir(input_dir)
    for file in files:
        file_path = os.path.join(input_dir, file)
        print(f'file_path = {file_path}')
        with open(file_path, 'r') as f:
            data = json.load(f)
            texts = data['texts']
            print(f'texts = {texts}')

        model_only_response = infer_model(texts)
        distance_response = infer_with_distance(texts)
        rewards, metrics = get_rewards(labels, [model_only_response, distance_response])
        print(rewards, metrics)
