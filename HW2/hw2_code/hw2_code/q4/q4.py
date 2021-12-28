# -*- coding: utf-8 -*-
"""q4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/116eBbb0Zr9dXU-B8ZReXZdREM0mV6uFe
"""

# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x),
                   axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    ## TODO
    # get A
    l2s = l2(test_datum.reshape(-1, 1).T, x_train)
    denom = np.sum(np.exp([(-d / (2 * (tau ** 2))) for d in l2s]))
    ai_s = np.array([(np.exp(-d / (2 * (tau ** 2))) / denom) for d in l2s])
    A = np.zeros((x_train.shape[0], x_train.shape[0]), float)
    np.fill_diagonal(A, ai_s)
    # get w star
    left = x_train.T @ A @ x_train + np.identity(x_train.shape[1]) * lam
    right = x_train.T @ A @ y_train.reshape(-1, 1)
    w_star = np.linalg.solve(left, right)
    # return y hat
    return test_datum.T @ w_star
    ## TODO


def run_validation(x, y, taus, val_frac):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    ## TODO
    x_validation = x[idx, :][:int(N * val_frac)]
    y_validation = np.take(y, idx)[:int(N * val_frac)]
    x_train = x[idx, :][int(N * val_frac):]
    y_train = np.take(y, idx)[int(N * val_frac):]
    train_loss = []
    validation_loss = []
    for t in taus:
        loss = []
        for i in range(len(x_train)):
            l = LRLS(x_train[i], x_train, y_train, t)[0]
            loss.append((l - y_train[i]) ** 2)
        train_loss.append(np.mean(loss))
        loss = []
        for i in range(len(x_validation)):
            l = LRLS(x_validation[i], x_train, y_train, t)[0]
            loss.append((l - y_validation[i]) ** 2)
        validation_loss.append(np.mean(loss))
    return train_loss, validation_loss
    ## TODO


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau
    # value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, validation_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(taus, validation_losses)
    plt.xlabel("Tau")
    plt.ylabel("Validation loss")
    plt.title("Tau vs Validation loss")
    plt.savefig('Q4.png')
    plt.show()