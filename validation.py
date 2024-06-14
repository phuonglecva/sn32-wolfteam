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
    scores = response.json()["result"]
    # print(f'model scores: {scores}')
    result = []
    for score in scores:
        if score < 0.5:
            result.append(False)
        elif score > 0.5:
            result.append(True)
        else:
            result.append(None)

    print(f'model results: {result}')
    # print(f'model count None: {result.count(None)}')
    # print(f'model fist half count None: {result[:150].count(None)}')
    # print(f'model second half count None: {result[150:].count(None)}')
    time_end = time.time_ns()
    print(f'time infer model: {(time_end - time_start) // 1000_000}')
    return result


def call_distance_api(texts, url=None):
    time_start = time.time_ns()
    import requests

    if url is None:
        url = "http://174.92.219.240:52109/predict"
    print(f'call distance api: {url}')
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        payload = json.dumps({
            "list_text": texts
        })
        response = requests.post(url, data=payload, headers=headers, timeout=360)
        time_end = time.time_ns()
        print(f'time processing distance of {len(texts)} sentences: {(time_end - time_start) // 1000_000} ms')
        return response.json()["result"]
    except Exception as e:
        print(f"Error: {e}")
        return [False] * len(texts)


def call_distance_api_multi_process(texts):
    time_start = time.time_ns()
    urls = [
        "http://174.92.219.240:52163/predict",
        "http://173.231.62.170:40042/predict",
        "http://174.92.219.240:52112/predict"
    ]
    import concurrent.futures
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_distance_api, texts, url) for url in urls]
        scores = [future.result() for future in futures]
        print(f'****** scores = {scores}')

    result = [min(scores[0][i], scores[1][i], scores[2][i]) for i in range(len(texts))]

    time_end = time.time_ns()
    print(f'time processing distance of {len(texts)} sentences: {(time_end - time_start) // 1000_000} ms')
    return result


def infer_with_distance(texts):
    print(f'test freelancers solution')
    distances = infer_distance(texts)
    preds = infer_model(texts)
    result = {i: None for i in range(len(texts))}
    preds_confs = {}
    for i in range(len(texts)):
        if distances[i] is not None:
            result[i] = distances[i]
    else:
        preds_confs[i] = preds[i]

    ai_count = len(list(filter(lambda x: x is True, result.values())))
    human_count = len(list(filter(lambda x: x is False, result.values())))

    sorted_preds_confs = sorted(preds_confs, key=preds_confs.get)

    for i in range(0, 150 - human_count):
        result[sorted_preds_confs[i]] = False

    for i in range(150 - human_count, len(sorted_preds_confs)):
        result[sorted_preds_confs[i]] = True
    return list(result.values())


def infer_distance(texts):
    try:
        print(f'start call infer_distance')
        time_start = time.time_ns()

        new_texts = []
        index_for_text = {}
        start_i = 0
        length_sentences = []
        for i, text in enumerate(texts):
            sentences = sent_tokenize(text)
            length_sentences.append(len(sentences))
            sentences = sentences[-4:]
            for _ in sentences:
                if i not in index_for_text:
                    index_for_text[i] = []
                index_for_text[i].append(start_i)
                start_i += 1
            new_texts.extend(sentences)

        print(f'length_sentences = {length_sentences}')
        import json
        with open('data.json', 'w') as f:
            json.dump(new_texts, f, indent=2)

        result = call_distance_api(new_texts)
        # result = call_distance_api_multi_process(new_texts)
        print(f'distance score: {result}')

        distance_result = []
        human_num_sentence_1_score = []
        ai_num_sentence_1_score = []

        human_num_sentence_2_score = []
        ai_num_sentence_2_score = []

        for i, text in enumerate(texts):
            list_index = index_for_text[i]
            list_result = [result[j] for j in list_index]
            if length_sentences[i] == 1:
                if i < 150:
                    human_num_sentence_1_score.append(list_result)
                else:
                    ai_num_sentence_1_score.append(list_result)
            if length_sentences[i] == 2:
                if i < 150:
                    human_num_sentence_2_score.append(list_result)
                else:
                    ai_num_sentence_2_score.append(list_result)

            count_hu = 0
            for score in list_result:
                if score < 0.1:
                    count_hu = count_hu + 1
            if count_hu == len(list_result):
                distance_result.append(False)
                continue

            if length_sentences[i] > 1:
                count_ai = 0
                for score in list_result:
                    if score > 0.2:
                        count_ai = count_ai + 1
                if count_ai > 1:
                    distance_result.append(True)
                    continue

            if length_sentences[i] == 2:
                if list_result[0] > 0.2 and list_result[1] < 0.1:
                    distance_result.append(False)
                    continue
                elif list_result[0] > 0.2 and list_result[1] > 0.2:
                    distance_result.append(True)
                    continue

            distance_result.append(None)

        print(f'human_num_sentence_1_score: {human_num_sentence_1_score}')
        print(f'ai_num_sentence_1_score: {ai_num_sentence_1_score}')
        print(f'human_num_sentence_2_score: {human_num_sentence_2_score}')
        print(f'ai_num_sentence_2_score: {ai_num_sentence_2_score}')

        print(f'distance result: {distance_result}')
        print_accuracy_distance(distance_result)

        time_end = time.time_ns()
        print(f'time process infer_distance of {len(texts)} sentences: {(time_end - time_start) // 1000_000} ms')

        return distance_result

    except Exception as e:
        print(f"Error full: {e}")
        return [False] * len(texts)


def infer_with_distance_backup(texts):
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


def print_accuracy_distance(response):
    first_half = response[:150]
    second_half = response[150:]
    first_wrong = first_half.count(True)
    second_wrong = second_half.count(False)
    count_not_none = [x is not None for x in response]
    print(f'accuracy_distance first_wrong is {first_wrong}')
    print(f'accuracy_distance second_wrong is {second_wrong}')
    print(f'count_not_none count_not_none is {count_not_none.count(True)}')


def cal_correct_prediction(response):
    first_half = response[:150].count(False)
    second_half = response[150:].count(True)
    return first_half + second_half


if __name__ == '__main__':
    import json
    import numpy as np
    import os
    from reward import get_rewards

    labels = [False] * 150 + [True] * 150
    input_dir = '/root/combine-method/sample_data'
    files = os.listdir(input_dir)
    sum_reward = [0, 0]
    sum_correct_pred = [0, 0]
    count = 0

    for file in files:
        file_path = os.path.join(input_dir, file)
        print(f'file_path = {file_path}')
        with open(file_path, 'r') as f:
            data = json.load(f)
            texts = data['texts']
            # print(f'texts = {texts}')

        model_only_response = infer_model(texts)
        # print(f'model only response: {model_only_response}')
        print_accuracy(model_only_response, 'model_only_response')

        distance_response = infer_with_distance(texts)
        print(f'distance response: {distance_response}')
        # print(f'distance response count None: {distance_response.count(None)}')
        # print(f'distance response first half count None: {distance_response[:150].count(None)}')
        # print(f'distance response second half count None: {distance_response[150:].count(None)}')

        print_accuracy(distance_response, 'distance_response')

        rewards, metrics = get_rewards(labels, [model_only_response, distance_response])
        print(rewards, metrics)

        sum_reward[0] += rewards[0]
        sum_reward[1] += rewards[1]
        count += 1
        print(
            f"=====> count = {count} Model reward AVG: {sum_reward[0] / count}, Combine reward AVG: {sum_reward[1] / count}", )

        sum_correct_pred[0] += cal_correct_prediction(model_only_response)
        sum_correct_pred[1] += cal_correct_prediction(distance_response)
        sum_text = count * 300
        print(
            f"------> count = {count} Model correct pred AVG: {sum_correct_pred[0] / sum_text}, Combine correct pred AVG: {sum_correct_pred[1] / sum_text}", )
