from utils import *

import numpy as np
import matplotlib.pyplot as plt
import vanilla_irt
from datetime import datetime


def question_acc(data):
    count = np.zeros(1774)
    correct = np.zeros(1774)
    for idx, j in enumerate(data['question_id']):
        count[j] += 1
        correct[j] += data['is_correct'][idx]
    count[count == 0] += 1
    return correct / count

def load_student_age_csv(data, path):
    # A helper function to load student age the csv file. Return 3 dictionary for 3 gender.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    birth_data = {
        "user_id": [],
        "birth_date": []
    }
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                birth_data["user_id"].append(int(row[0]))
                birth_data["birth_date"].append(
                    datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S.%f").year
                    if row[2] != "" else -1)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass

    age_0 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    age_1 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    age_2 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    for i, id in enumerate(data["user_id"]):
        if id in birth_data["user_id"]:
            if birth_data["birth_date"][birth_data["user_id"].index(id)] == -1:
                age_0["user_id"].append(id)
                age_0["question_id"].append(data["question_id"][i])
                age_0["is_correct"].append(data["is_correct"][i])
            elif birth_data["birth_date"][
                birth_data["user_id"].index(id)] < 2005:
                age_1["user_id"].append(id)
                age_1["question_id"].append(data["question_id"][i])
                age_1["is_correct"].append(data["is_correct"][i])
            else:
                age_2["user_id"].append(id)
                age_2["question_id"].append(data["question_id"][i])
                age_2["is_correct"].append(data["is_correct"][i])

    return age_0, age_1, age_2


def load_student_gender_csv(data, path):
    # A helper function to load student gender the csv file. Return 3 dictionary for 3 gender.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    gender_data = {
        "user_id": [],
        "gender": []
    }
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                gender_data["user_id"].append(int(row[0]))
                gender_data["gender"].append(int(row[1]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    gender_0 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    gender_1 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    gender_2 = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for i, id in enumerate(data["user_id"]):
        if id in gender_data["user_id"]:
            if gender_data["gender"][gender_data["user_id"].index(id)] == 0:
                gender_0["user_id"].append(id)
                gender_0["question_id"].append(data["question_id"][i])
                gender_0["is_correct"].append(data["is_correct"][i])
            elif gender_data["gender"][gender_data["user_id"].index(id)] == 1:
                gender_1["user_id"].append(id)
                gender_1["question_id"].append(data["question_id"][i])
                gender_1["is_correct"].append(data["is_correct"][i])
            else:
                gender_2["user_id"].append(id)
                gender_2["question_id"].append(data["question_id"][i])
                gender_2["is_correct"].append(data["is_correct"][i])
    return gender_0, gender_1, gender_2

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha, c):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0
    user_id = data['user_id']
    question_id = data['question_id']
    C = data['is_correct']
    for idx in range(len(user_id)):
        i = user_id[idx]
        j = question_id[idx]
        e = np.exp(alpha[j] * (theta[i] - beta[j]))
        log_lklihood += C[idx] * np.log(c[j] + e) - np.log(1 + e) + (1 - C[idx]) * np.log(1 - c[j])
    return -log_lklihood


def update_params(data, lr, theta, beta, alpha, c, i):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    user_id = data['user_id']
    question_id = data['question_id']
    C = data['is_correct']
    d_c = np.zeros(1774)
    # lr = (1 / 2) ** int(i / 10) * lr
    for idx in range(len(user_id)):
        i = user_id[idx]
        j = question_id[idx]
        e = np.exp(alpha[j] * (theta[i] - beta[j]))
        theta[i] += lr * e * alpha[j] * (C[idx] / (c[j] + e) - 1 / (1 + e))
        beta[j] += lr * e * alpha[j] * (-1 * C[idx] / (c[j] + e) + 1 / (1 + e))
        alpha[j] += lr * e * (theta[i] - beta[j]) * (C[idx] / (c[j] + e) - 1 / (1 + e))
        # d_c[j] += lr * (C[idx] / (c[j] + e) - (1 - C[idx]) / (1 - c[j]))
    # c = sigmoid(c + d_c) / 3
    return theta, beta, alpha, c


def irt(data, val_data, test_data, lr, iterations, theta, beta, alpha, c, printtest=False):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # alpha = np.ones(1774) 
    # c = np.zeros(1774) + pseudo_guess
    # theta = np.zeros(542)
    # beta = np.zeros(1774)
    val_acc_lst = []
    train_nlld_lst = []
    for i in range(iterations):
        theta, beta, alpha, c = update_params(data, lr, theta, beta, alpha, c, i)
        train_neg_lld = neg_log_likelihood(data, theta, beta, alpha, c)
        train_nlld_lst.append(train_neg_lld)
        train_score = evaluate(data, theta, beta, alpha, c)
        val_score = evaluate(val_data, theta, beta, alpha, c)
        val_acc_lst.append(val_score)
        test_score = evaluate(test_data, theta, beta, alpha, c)
        if printtest:
            print("Iteration: {} \t NLLK: {:.12f} \t Train Score: {:.16f} \t Val Score: {:.16f} \t Test Score: {:.16f}".\
            format(i, train_neg_lld, train_score, val_score, test_score))
        else: print("Iteration: {} \t NLLK: {:.12f} \t Train Score: {:.16f} \t Val Score: {:.16f}".\
            format(i, train_neg_lld, train_score, val_score))
    return theta, beta, alpha, c, train_nlld_lst, val_acc_lst


def evaluate(data, theta, beta, alpha, c):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = c[q] + (1 - c[q]) * sigmoid(x)
        # p_a = c[q] + (1 - c[q]) / (1 + np.exp(-1 * alpha[q] * (theta[u] - beta[q])))
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def _eval(i, j, theta, beta, alpha, c):
    x = (alpha[j] * (theta[i] - beta[j])).sum()
    p = c[j] + (1 - c[j]) * sigmoid(x)
    return p >= 0.5


def train_model(train_data, test_data, lr, iterations, pseudo_guess):
    path = "../data/student_meta.csv"
    train_data_0, train_data_1, train_data_2 = load_student_gender_csv(train_data, path)
    test_data_0, test_data_1, test_data_2 = load_student_gender_csv(test_data, path)
    theta = np.zeros(542)
    beta = np.zeros(1774)
    alpha = np.ones(1774) 
    c = np.zeros(1774) + pseudo_guess
    theta_0, beta_0, alpha_0, c_0, *_ = irt(train_data, test_data_0, None, lr, iterations, theta, beta, alpha, c)
    theta_1, beta_1, alpha_1, c_1, *_ = irt(train_data_1, test_data_1, None, lr, iterations, theta, beta, alpha, c)
    theta_2, beta_2, alpha_2, c_2, *_ = irt(train_data_2, test_data_2, None, lr, iterations, theta, beta, alpha, c)
    pred = []
    for idx, i in enumerate(test_data['user_id']):
        j = test_data['question_id'][idx]
        if i in test_data_1['user_id']:
            pred.append(_eval(i, j, theta_1, beta_1, alpha_1,c_1))
        elif i in test_data_2['user_id']:
            pred.append(_eval(i, j, theta_2, beta_2, alpha_2, c_2))
        else:
            pred.append(_eval(i, j, theta_0, beta_0, alpha_0, c_0))
    return np.sum((test_data["is_correct"] == np.array(pred))) \
           / len(test_data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.014
    iterations = 39
    pseudo_guess = 0.1
    # print(train_model(train_data, val_data, lr, iterations, pseudo_guess))
    theta = np.zeros(542)
    beta = np.zeros(1774)
    alpha = np.ones(1774) 
    c = np.zeros(1774) + pseudo_guess
    theta, beta, alpha, c, _, val_acc_lst_1 = irt(train_data, val_data, test_data, lr, iterations, theta, beta, alpha, c, printtest=True)
    _, _, val_acc_lst_2, *_ = vanilla_irt.irt(train_data, val_data, lr, iterations)
    plt.plot(np.arange(iterations), val_acc_lst_1, label='3 param')
    plt.plot(np.arange(iterations), val_acc_lst_2, label='1 param')
    plt.legend()
    plt.savefig('compare.png')
    plt.show()
    print(f"theta has mean {theta.mean()} median {np.median(theta)} variance {np.var(theta)}")
    print(f"beta has mean {beta.mean()} median {np.median(beta)} variance {np.var(beta)}")
    print(f"alpha has mean {alpha.mean()} median {np.median(alpha)} variance {np.var(alpha)}")
    print(f"c has mean {c.mean()} median {np.median(c)} variance {np.var(c)}")
    print(f'the final training accuracy is {evaluate(train_data, theta, beta, alpha, c)}')
    print(f'the final validation accuracy is {evaluate(val_data, theta, beta, alpha, c)}')
    print(f'the final test accuracy is {evaluate(test_data, theta, beta, alpha, c)}')
    # plt.subplot(1, 2, 1)
    # plt.title('training')
    # plt.plot(np.arange(iterations), train_nlld_lst)
    # plt.subplot(1, 2, 2)
    # plt.title('validation')
    # plt.plot(np.arange(iterations), val_nlld_lst)
    # plt.savefig('q2b.png')
    # plt.show()
    # plt.clf()
    # np.random.seed(0)
    # # j_s = np.random.randint(1774, size=3)
    # j_s = [0, 44, 24]
    # theta = np.sort(theta)
    # for j in j_s:
    #     plt.plot(theta, c[j] + (1 - c[j]) * sigmoid(alpha[j] * (theta - beta[j])))
    # # plt.savefig('q2d.png')
    # plt.show()

    # kaggle_data = load_private_test_csv(root_dir="../data")
    # pred = []
    # for i, q in enumerate(kaggle_data["question_id"]):
    #     u = kaggle_data["user_id"][i]
    #     x = (alpha[q] * (theta[u] - beta[q])).sum()
    #     pred.append((c[q] + (1 - c[q]) * sigmoid(x)) >= 0.5)
    # kaggle_data['is_correct'] = pred
    # save_private_test_csv(kaggle_data)

    # q_acc = question_acc(train_data)
    # print(np.mean(q_acc[q_acc > 0]))
    # print(np)
    # print(np.mean(quesnp.min(question_acc(train_data)), np.argmin(question_acc(train_data)))
    
   


if __name__ == "__main__":
    main()
