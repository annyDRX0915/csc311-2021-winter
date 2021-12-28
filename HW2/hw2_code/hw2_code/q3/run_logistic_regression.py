from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 100
    }
    weights = np.zeros(M + 1).reshape(-1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    ce_t = [evaluate(train_targets, logistic_predict(weights, train_inputs))[0]]
    ce_v = [evaluate(valid_targets, logistic_predict(weights, valid_inputs))[0]]
    it_s = [0]
    acc_t = [
        evaluate(train_targets, logistic_predict(weights, train_inputs))[0]]
    acc_v = [
        evaluate(valid_targets, logistic_predict(weights, valid_inputs))[1]]
    for t in range(hyperparameters["num_iterations"]):
        df = logistic(weights, train_inputs, train_targets, hyperparameters)[1]
        weights -= hyperparameters["learning_rate"] * df
        ct, at = evaluate(train_targets,
                          logistic_predict(weights, train_inputs))
        cv, av = evaluate(valid_targets,
                          logistic_predict(weights, valid_inputs))
        ce_t.append(ct)
        ce_v.append(cv)
        acc_t.append(at)
        acc_v.append(av)
        it_s.append(t + 1)

    test_inputs, test_targets = load_test()
    tr_result = evaluate(train_targets, logistic_predict(weights, train_inputs))
    val_result = evaluate(valid_targets,
                          logistic_predict(weights, valid_inputs))
    te_result = evaluate(test_targets, logistic_predict(weights, test_inputs))
    print("Train")
    print("CE: " + str(tr_result[0]) + "classification error: " + str(
        1 - tr_result[1]))
    print("Validation")
    print("CE: " + str(val_result[0]) + "classification error: " + str(
        1 - val_result[1]))
    print("Test")
    print("CE: " + str(te_result[0]) + "classification error: " + str(
        1 - te_result[1]))

    plt.plot(it_s, ce_t, label="train loss", c='b')
    plt.plot(it_s, ce_v, label="validation loss", c='r')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy")
    plt.title("Iteration vs cross entropy")
    plt.savefig("Q3.2.png")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
