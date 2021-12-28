import numpy as np
import matplotlib.pyplot as plt


def question1():
    # preset the mean and standard deviation
    means = []
    std_dev = []
    x_axis = []
    # for 0 - 10
    for i in range(11):
        print(i)
        # generate the data
        data = np.random.rand(100, 2 ** i)
        # preset the euclidean distances
        euc_dists = []
        # calculate the euclidean distances
        for ptr1 in range(100):
            for ptr2 in range(ptr1 + 1, 100):
                euc_dist = 0.0
                for pos in range(2 ** i):
                    euc_dist += (data[ptr1][pos] - data[ptr2][pos]) ** 2
                euc_dists.append(euc_dist)
        means.append(np.mean(np.array(euc_dists)))
        std_dev.append(np.std(np.array(euc_dists)))
        x_axis.append(2 ** i)
    print(means)
    print(std_dev)
    print(x_axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax1.plot(x_axis, means, marker='o')
    ax2.plot(x_axis, std_dev, marker='*')
    ax1.set_title('Dimension vs Means')
    ax2.set_title('Dimension vs std dev')
    plt.savefig('graphs.png')
    plt.show()


if __name__ == '__main__':
    print("Question 1")
    question1()
