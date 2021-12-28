'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import math


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for d in range(10):
        datas = data.get_digits_by_label(train_data, train_labels, d)
        means[d,] = np.mean(datas, axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for d in range(10):
        datas = data.get_digits_by_label(train_data, train_labels, d)
        mean = means[d]
        cov = (datas - mean).T @ (datas - mean)
        covariances[d, :, :] = (cov / len(datas)) + np.identity(64) * 0.01
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    l_ls = np.zeros((len(digits), 10))
    for i in range((len(digits))):
        data = digits[i, :]
        for d in range(10):
            mean = means[d, :]
            cov = covariances[d, :, :]
            cov_inverse = np.linalg.inv(cov)
            d_m = data - mean
            l_ls[i][d] = (-64 / 2) * np.log(2 * np.pi) + \
                         (-1 / 2) * np.log(np.linalg.det(cov)) + \
                         (-1 / 2) * d_m @ cov_inverse @ d_m.T
    return l_ls


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    g_l_ls = generative_likelihood(digits, means, covariances)
    l_ls = np.zeros((len(digits), 10))
    for i in range(len(digits)):
        g_l_l = g_l_ls[i, :]
        for d in range(10):
            computed_result = g_l_l[d] + math.log(1 / 10) - \
                              math.log(np.sum(np.exp(g_l_l)) / 10)
            l_ls[i][d] = computed_result
    return l_ls


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    result = 0
    for i in range(len(digits)):
        result += cond_likelihood[i][int(labels[i])]
    result = result / len(digits)
    return result


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data(
        'data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # a
    print('Avg conditional log-likelihood on training set is: ' +
          str(avg_conditional_likelihood(train_data, train_labels, means,
                                         covariances)))
    print('Avg conditional log-likelihood on testing set is: ' +
          str(avg_conditional_likelihood(test_data, test_labels, means,
                                         covariances)))

    # b
    train_pred = classify_data(train_data, means, covariances)
    test_pred = classify_data(test_data, means, covariances)
    print("Accuracy on training set is: " +
          str(np.count_nonzero(train_pred == train_labels) / len(train_data)))
    print("Accuracy on training set is: " +
          str(np.count_nonzero(test_pred == test_labels) / len(test_data)))

    # c
    for i in range(10):
        e_val, e_vec = np.linalg.eig(covariances[i])
        plt.imshow(e_vec[:, np.argmax(e_val)].reshape(8, 8))
        plt.savefig('./{}.png'.format(i))


if __name__ == '__main__':
    main()
