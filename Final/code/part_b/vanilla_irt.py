from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    l = lambda i : is_correct[i] * np.log(sigmoid(theta[user_id[i]] - beta[question_id[i]])) + \
                (1 - is_correct[i]) * np.log(1 - sigmoid(theta[user_id[i]] - beta[question_id[i]]))
    log_lklihood = np.sum([l(i) for i in range(len(user_id))])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    pass
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    d_theta = lambda i : is_correct[i] - sigmoid(theta[user_id[i]] - beta[question_id[i]])
    d_beta = lambda i : sigmoid(theta[user_id[i]] - beta[question_id[i]]) - is_correct[i]
    for i in range(len(user_id)):
        theta[user_id[i]] += lr * d_theta(i)
        beta[question_id[i]] += lr * d_beta(i)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
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
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)
    # theta = np.zeros(542)
    # beta = np.zeros(1774)
    val_acc_lst = []
    train_nlld_lst = []
    val_nlld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_nlld_lst.append(train_neg_lld)
        val_nlld_lst.append(val_neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("Iteration: {} \t NLLK: {:.12f} \t Score: {:.16f}".format(i, train_neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nlld_lst, val_nlld_lst


def evaluate(data, theta, beta):
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
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(len(train_data['user_id']))
    print(len(val_data['user_id']))

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 40
    theta, beta, val_acc_lst, train_nlld_lst, val_nlld_lst = irt(train_data, val_data, lr, iterations)
    print(f'the final training accuracy is {evaluate(train_data, theta, beta)}')
    print(f'the final validation accuracy is {val_acc_lst[-1]}')
    print(f'the final test accuracy is {evaluate(test_data, theta, beta)}')
    plt.subplot(1, 2, 1)
    plt.title('training')
    plt.plot(np.arange(iterations), train_nlld_lst)
    plt.subplot(1, 2, 2)
    plt.title('validation')
    plt.plot(np.arange(iterations), val_nlld_lst)
    plt.savefig('q2b.png')
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plt.clf()
    np.random.seed(0)
    theta = np.sort(theta)
    for j in np.random.randint(1774, size=3):
        plt.plot(theta, sigmoid(theta - beta[j]))
    plt.savefig('q2d.png')
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
