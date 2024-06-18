# Description: This file contains the code for the validation of the model
import time

import requests
from nltk.tokenize import sent_tokenize
import json
from app_config import AppConfig

APP_CONFIG = AppConfig()


def infer_model(texts):
    print(f'start infer_model of {len(texts)} texts')
    time_start = time.time_ns()
    url = APP_CONFIG.get_model_server_url()
    payload = {
        "list_text": texts
    }
    response = requests.request("POST", url, json=payload, timeout=APP_CONFIG.get_model_timeout())
    scores = response.json()["result"]
    # print(f'model scores: {scores}')
    result = []
    for score in scores:
        if score < 0.5:
            result.append(False)
        elif score >= 0.5:
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


def call_distance_api(sentences, url=None):
    time_start = time.time_ns()
    print(f'call distance api: {url}, len(sentences) = {len(sentences)}')
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        payload = json.dumps({
            "list_text": sentences
        })
        response = requests.post(url, data=payload, headers=headers, timeout=APP_CONFIG.get_nns_timeout())
        time_end = time.time_ns()
        print(f'time processing distance of {len(sentences)} sentences: {(time_end - time_start) // 1000_000} ms')
        return response.json()["result"]
    except Exception as e:
        print(f"Error: {e}")
        return [-1] * len(sentences)


def call_distance_api_multi_process(texts, sentences):
    print(f'start call_distance_api_multi_process len(texts) = {len(texts)}, len(sentences) = {len(sentences)}')
    time_start = time.time_ns()
    urls = APP_CONFIG.get_all_nns_server_url()
    import concurrent.futures
    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_distance_api_and_model_api, texts, sentences, url) for url in urls]
        scores = [future.result() for future in futures]
        # print(f'****** scores = {scores}')

    time_end = time.time_ns()
    print(f'time processing distance of {len(sentences)} sentences: {(time_end - time_start) // 1000_000} ms')
    return scores


def predict_texts(texts, validator_hotkey=None):
    APP_CONFIG.load_app_config()
    len_texts = len(texts)
    if len_texts == 300:
        return infer_with_distance(texts, validator_hotkey)
    else:
        return infer_model(texts)


def call_distance_api_and_model_api(texts, sentences, url):
    if url is not None and len(url) > 1:
        return call_distance_api(sentences, url)
    else:
        return infer_model(texts)


def get_distance_result(index_for_text, result, length_sentences, texts):
    distance_result = []

    human_threshold = APP_CONFIG.get_nns_hu_threshold()
    ai_threshold = APP_CONFIG.get_nns_ai_threshold()
    for i, text in enumerate(texts):
        list_index = index_for_text[i]
        list_result = [result[j] for j in list_index]
        count_hu = 0
        for score in list_result:
            if score < human_threshold:
                count_hu = count_hu + 1
        if count_hu == len(list_result):
            distance_result.append(False)
            continue

        if length_sentences[i] > 1:
            count_ai = 0
            for score in list_result:
                if score > ai_threshold:
                    count_ai = count_ai + 1
            if count_ai > 1:
                distance_result.append(True)
                continue

        if length_sentences[i] == 2:
            if list_result[0] > ai_threshold and list_result[1] < human_threshold:
                distance_result.append(False)
                continue
            elif list_result[0] > ai_threshold and list_result[1] > ai_threshold:
                distance_result.append(True)
                continue

        distance_result.append(None)

    print(f'distance result: {distance_result}')

    num_true = distance_result.count(True)
    num_false = distance_result.count(False)
    count_not_none = [x is not None for x in distance_result]
    print(f'print_accuracy_distance_finney num True {num_true}')
    print(f'print_accuracy_distance_finney num False {num_false}')
    print(f'count_not_none count_not_none is {count_not_none.count(True)}')
    if 10 < num_true < 175:
        return True, num_true
    else:
        return False, -1


def infer_with_distance(texts, validator_hotkey=None):
    distances = infer_distance(texts, validator_hotkey)
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
    print(f'human_count = {human_count}, ai_count = {ai_count}')
    if ai_count > 150 or human_count > 150:
        return preds

    sorted_preds_confs = sorted(preds_confs, key=preds_confs.get)
    print(f'sorted_preds_confs len = {len(sorted_preds_confs)}, sorted_preds_confs: {sorted_preds_confs}')

    for i in range(0, 150 - human_count):
        result[sorted_preds_confs[i]] = False

    for i in range(150 - human_count, len(sorted_preds_confs)):
        result[sorted_preds_confs[i]] = True
    return list(result.values())


def get_url_by_validator_hotkey(validator_hotkey):
    return APP_CONFIG.get_nns_server_url(validator_hotkey)


def infer_distance(texts, validator_hotkey=None):
    try:
        print(f'start call infer_distance of {len(texts)}, validator hotkey = {validator_hotkey}')

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

        # result = call_distance_api(new_texts, url)
        urls = APP_CONFIG.get_all_nns_server_url()
        scores = call_distance_api_multi_process(texts, new_texts)
        print(f'distance score len: {len(scores)}')
        hit_pile = []
        for i in range(len(scores)):
            result = scores[i]
            hit, num_true = get_distance_result(index_for_text, result, length_sentences, texts)
            if hit:
                hit_pile.append(f'{i + 1}_{num_true}_{urls[i]}')

        print(f'Hit file {validator_hotkey}:{hit_pile}')

    except Exception as e:
        print(f"Error full: {e}")
        return [None] * len(texts)


# def infer_with_distance_backup(texts):
#     distances = infer_distance(texts)
#     preds = infer_model(texts)
#     result = []
#     for i in range(len(distances)):
#         if distances[i] is not None:
#             result.append(distances[i])
#         else:
#             result.append(preds[i])
#     return result

def print_accuracy(response, prefix):
    first_half = response[:150]
    second_half = response[150:]
    predict_correct = first_half.count(False) + second_half.count(True)
    accuracy = predict_correct / 300
    print(f'{prefix} accuracy is {accuracy}')


def print_accuracy_distance_finney(response):
    num_true = response.count(True)
    num_false = response.count(False)
    count_not_none = [x is not None for x in response]
    print(f'print_accuracy_distance_finney num True {num_true}')
    print(f'print_accuracy_distance_finney num False {num_false}')
    print(f'count_not_none count_not_none is {count_not_none.count(True)}')


def print_accuracy_distance_test(response):
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

        distance_response = infer_with_distance(texts, "5CXRfP2ekFhe62r7q3vppRajJmGhTi7vwvb2yr79jveZ282w")
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
