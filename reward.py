import traceback
from typing import List
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score


def reward(y_pred: np.array, y_true: np.array) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    preds = np.round(y_pred)

    # accuracy = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, preds)
    ap_score = average_precision_score(y_true, y_pred)

    res = {'fp_score': 1 - fp / len(y_pred),
           'f1_score': f1,
           'ap_score': ap_score}
    reward = sum([v for v in res.values()]) / len(res)
    return reward, res


def count_penalty(
        y_pred: np.array,
        check_predictions: np.array,
        check_ids: np.array,
        version_predictions_array: List
) -> float:
    bad = np.any((y_pred < 0) | (y_pred > 1))

    print(f'check_predictions check_predictions = {check_predictions}')
    print(f'y_pred y_pred = {y_pred}')
    print(f'check_ids check_ids = {check_ids}')
    if (check_predictions.round(2) != y_pred[check_ids].round(2)).any():
        bad = 1

    if version_predictions_array:
        bad = 1

    return 0 if bad else 1


def get_rewards(
        labels: np.array,
        predictions_list: List[List[bool]],
        check_predictions_list: List[List[bool]],
        version_predictions_list: List[List[bool]],
        check_ids: np.array
):
    rewards = []
    metrics = []
    for uid in range(len(predictions_list)):
        try:
            if not predictions_list[uid] or len(predictions_list[uid]) != len(labels) or \
                    not check_predictions_list[uid] or len(check_predictions_list[uid]) != len(check_ids):
                rewards.append(0)
                metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})
                print(f'have error with checked or version in rewards function')
                continue

            predictions_array = np.array(predictions_list[uid])
            check_predictions_array = np.array(check_predictions_list[uid])

            miner_reward, metric = reward(predictions_array, labels)
            penalty = count_penalty(
                predictions_array, check_predictions_array, check_ids, version_predictions_list[uid])

            miner_reward *= penalty
            rewards.append(miner_reward)
            metric['penalty'] = penalty
            metrics.append(metric)
        except Exception as e:
            print(f'error {e}')
            metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})
            traceback.print_exc()

    return rewards, metrics
