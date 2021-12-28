"""
CSC311 Homework 1 2021 Fall
Anny Runxuan Dai
UTorid: daianny
student number: 1004881933
"""

import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import numpy as np


def load_data():
    """
    load the data and splits the entire dataset randomly into
    70% training, 15% validation, and 15% test examples
    :return: the data and labels
    """
    data = []
    label = []
    # read from clean_fake and append the label
    with open('clean_fake.txt') as f:
        line = f.readline()
        while line:
            data.append(line.strip())
            label.append(0)
            line = f.readline()
    # read from clean_real and append the label
    with open('clean_real.txt') as f:
        line = f.readline()
        while line:
            data.append(line.strip())
            label.append(1)
            line = f.readline()
    # vectorize the data
    vectorizer = CountVectorizer()
    # only the vocabulary from data_train is needed to train
    data = vectorizer.fit_transform(data)
    # split the data
    data_tr, data, label_tr, label = \
        train_test_split(data, label, train_size=0.7, random_state=0)
    data_val, data_te, label_val, label_te = \
        train_test_split(data, label, train_size=0.5, random_state=0)
    label_tr = np.array(label_tr)
    label_te = np.array(label_te)
    label_val = np.array(label_val)
    # return the data
    return data_tr, data_val, data_te, \
           label_tr, label_val, label_te, vectorizer


def select_model(data_tr, data_val, data_te, label_tr,
                 label_val, label_te):
    """
     trains the decision tree classifier using at least 5 different values
     of max_depth, as well as two different split criteria (information gain
     and Gini coefficient), evaluates the performance of each one on the
     validation set, and prints the resulting accuracies of each model.
    :param data_tr: the training data
    :param data_val: the validation data
    :param data_te: the testing data
    :param label_tr: the training labels
    :param label_val: the validation labels
    :param label_te: the testing labels
    :param vectorizer: the vectorizer of these data
    :return: the best tree
    """
    # the array to save the trees
    clfs = []
    # construct the trees
    for i in range(5, 31, 5):
        clf_gini = tree.DecisionTreeClassifier(max_depth=i, criterion="gini",
                                               random_state=0)
        clf_gini.fit(data_tr, label_tr)
        clfs.append(clf_gini)
        clf_ig = tree.DecisionTreeClassifier(max_depth=i, criterion="entropy",
                                             random_state=0)
        clf_ig.fit(data_tr, label_tr)
        clfs.append(clf_ig)
    # find the best tree with the best score
    # and print the accuracies using validation set
    best_tree = None
    best_score = 0
    print("accuracies of validation set")
    for i in range(10):
        predict_val = clfs[i].predict(data_val)
        score = np.count_nonzero(predict_val == label_val) / len(predict_val)
        if score > best_score:
            best_tree = clfs[i]
            best_score = score
        print(score)
    print()
    # print the accuracies using the testing sets
    print("accuracies of testing set")
    for i in range(10):
        predict_te = clfs[i].predict(data_te)
        print(np.count_nonzero(predict_te == label_te) / len(predict_te))
    print()
    return best_tree


def compute_information_gain(word, threshold, vectorizer, data_tr, label_tr):
    """
    return the IG(Y|x) on the training set
    :param word: the word to split
    :param threshold: the threshold of the word
    :param vectorizer: the vectorizer of the data
    :param data_tr: the training data
    :param label_tr: the training labels
    :return: the IG(Y|x)
    """
    # find the index of the word in the dataset
    word_index = None
    if word in vectorizer.vocabulary_:
        word_index = vectorizer.vocabulary_[word]
    else:
        return 0.0
    # combine the data and the training set and make them numpy
    data_tr = np.array(data_tr.toarray())
    training = np.c_[data_tr, label_tr]
    # split the data using the word and threshold
    less_set = training[training[:, word_index] <= threshold]
    greater_set = training[training[:, word_index] > threshold]
    # calculate H(Y)
    hy = entropy(np.count_nonzero(training[:, -1] == 1), len(training)) + \
         entropy(np.count_nonzero(training[:, -1] == 0), len(training))
    # calculate H(Y|x)
    hyx = (len(less_set) / len(training)) * (
            entropy(np.count_nonzero(less_set[:, -1] == 1), len(less_set)) +
            entropy(np.count_nonzero(less_set[:, -1] == 0), len(less_set))) + \
          (len(greater_set) / len(training)) * (
                  entropy(np.count_nonzero(greater_set[:, -1] == 1),
                          len(greater_set))
                  + entropy(np.count_nonzero(greater_set[:, -1] == 0),
                            len(greater_set)))
    # return IG(Y|x)
    return hy - hyx


def entropy(num, tot):
    if num / tot == 0.0:
        return 0
    return - ((num / tot) * math.log((num / tot), 2))


if __name__ == '__main__':
    # load the data
    data_tr, data_val, data_te, label_tr, label_val, label_te, \
    vectorizer = load_data()
    best_tree = select_model(data_tr, data_val, data_te, label_tr, label_val,
                             label_te)
    # export the dot file of the best tree
    dot_data = tree.export_graphviz(best_tree, out_file="tree.dot", filled=True,
                                    max_depth=2,
                                    feature_names=
                                    vectorizer.get_feature_names_out())
    # calculate the information gain and print them
    print("information gain of 'the': " +
          str(compute_information_gain("the", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'hillary': " +
          str(compute_information_gain("hillary", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'donald': " +
          str(compute_information_gain("donald", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'trump': " +
          str(compute_information_gain("trump", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'clean': " +
          str(compute_information_gain("clean", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'decided': " +
          str(compute_information_gain("decided", 0.5, vectorizer,
                                       data_tr, label_tr)))
    print("information gain of 'gun': " +
          str(compute_information_gain("gun", 0.5, vectorizer,
                                       data_tr, label_tr)))
