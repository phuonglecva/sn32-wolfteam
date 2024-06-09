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
        length_sentences = []
        for i, text in enumerate(texts):
            sentences = sent_tokenize(text)
            length_sentences.append(len(sentences))
            sentences = sentences[-3:]
            for _ in sentences:
                if i not in index_for_text:
                    index_for_text[i] = []
                index_for_text[i].append(start_i)
                # print(index_for_text)
                start_i += 1
            new_texts.extend(sentences)
        # write text to file data.json
        # print(f'length_sentences = {length_sentences}')
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
        print(f'distance score: {result}')
        distance_result = []
        for i, text in enumerate(texts):
            list_index = index_for_text[i]
            list_result = [result[j] for j in list_index]
            is_human = True
            for score in list_result:
                is_human = is_human and (score < 0.001)
            if is_human:
                distance_result.append(False)
                continue

            if length_sentences[i] > 1:
                is_ai = True
                for score in list_result:
                    is_ai = is_ai and (score > 0.1)
                if is_ai:
                    distance_result.append(True)
                    continue

            distance_result.append(None)

        print(f'distance result: {distance_result}')
        time_end = time.time_ns()
        print(f'time processing distance: {(time_end - time_start) // 1000_000} ms')
        print_accuracy(distance_result, 'distance_result')

        return distance_result

    except Exception as e:
        print(f"Error full: {e}")
        return [False] * len(texts)


def infer_with_distance(texts):
    distances = infer_distance(texts)
    preds = infer_model(texts)
    result = []
    for i in range(len(distances)):
        if distances[i] is not None:
            result.append(distances[i])
        else:
            result.append(preds[i])
    return result


def print_accuracy(response, prefix):
    first_half = response[:150]
    second_half = response[150:]
    predict_correct = first_half.count(False) + second_half.count(True)
    accuracy = predict_correct / 300
    print(f'{prefix} accuracy is {accuracy}')


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
            # print(f'texts = {texts}')

        model_only_response = infer_model(texts)
        print(f'model only response: {model_only_response}')
        print_accuracy(model_only_response, 'model_only_response')

        distance_response = infer_with_distance(texts)
        print(f'distance response: {distance_response}')
        print_accuracy(distance_response, 'distance_response')

        rewards, metrics = get_rewards(labels, [model_only_response, distance_response])
        print(rewards, metrics)
