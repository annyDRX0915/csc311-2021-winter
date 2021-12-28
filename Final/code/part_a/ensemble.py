# TODO: complete this file.

from utils import *
from item_response import irt, sigmoid

import numpy as np
import matplotlib.pyplot as plt

def resample_csv(data):
    """
    create a bagging of csv data

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: A re-shuffled dictionary
    """
    resample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    n = len(data['user_id'])
    for i in np.random.choice(n, n):
        resample['user_id'].append(data['user_id'][i])
        resample['question_id'].append(data['question_id'][i])
        resample['is_correct'].append(data['is_correct'][i])
    return resample

def evaluate(data, params_lst):
    """ Evaluate the average of models (by majority vote) given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param params_lst: list of tuple of (theta, beta)
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        count = 0
        for theta, beta in params_lst:
            u = data["user_id"][i]
            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x)
            count += int(p_a >= 0.5)
        pred.append(count)
    pred = np.array(pred) >= 2
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    resample1 = resample_csv(train_data)
    resample2 = resample_csv(train_data)
    resample3 = resample_csv(train_data)
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # use the same tuned hyper parameter
    lr = 0.01
    iterations = 40
    theta1, beta1, *_ = irt(resample1, val_data, lr, iterations)
    theta2, beta2, *_ = irt(resample2, val_data, lr, iterations)
    theta3, beta3, *_ = irt(resample3, val_data, lr, iterations)
    print(evaluate(val_data, [(theta1, beta1), (theta2, beta2), (theta3, beta3)]))
    print(evaluate(test_data, [(theta1, beta1), (theta2, beta2), (theta3, beta3)]))


if __name__ == '__main__':
    main()

