#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:17:09 2018

@author: fantasie
"""

import pickle
import numpy as np
import keras
from glob import glob
import matplotlib.pyplot as plt


def load_pkl(filename, data_dir='../data/'):
    """
    Load statistics from pkl files
    :param filename: file name of the stats cache
    :param data_dir: directory of the cache file
    :return: stats in variables. The variables are different for different cache files.
    """
    if 'hist' in filename:
        hist_test, hist_train, \
            raw_test, raw_train, \
            stats_test, stats_train = pickle.load(open(data_dir + filename, "rb"))
        return hist_test, hist_train, raw_test, raw_train, stats_test, stats_train
    elif filename == 'stats_25_train_test.pkl':
        stats_test, stats_train = pickle.load(open(data_dir + filename, "rb"))
        return stats_test, stats_train
    elif filename == 'stats_25.pkl' or filename == 'stats.pkl':
        raw, stats = pickle.load(open(data_dir + filename, "rb"))
        return raw, stats
    elif 'online' in filename:
        cumulative_stats_train, cumulative_hist_train = pickle.load(open(data_dir + filename, "rb"))
        return cumulative_stats_train, cumulative_hist_train
    else:
        return None


def compute_cumulative_confusion_matrix(hists, ys):
    """
    Compute the cumulative confusion matrix from multi-runs
    :param hists: array of size N x C, where N is the #samples and C is the #classes
              each row shows #counts for the sample being classified to each class
    :param ys: ground truth with size (N,)
    :return: confusion matrix, an array of size C x C
    """
    assert len(hists) == len(ys)

    confusion_matrix = np.zeros((hists.shape[1], hists.shape[1]))
    for i in range(len(hists)):
        h, y = hists[i], ys[i]
        for j in range(len(h)):
            confusion_matrix[y, j] += h[j]

    return confusion_matrix.astype(int)


def relabel_training_samples(hists, labels, conf_threshold=5, gt_threshold=1):
    """
    Naive implementation of re-labeling the ground truth according to predictions for it during training.
    If the sample sample is seldom or never classified correctly during training, we assume it belongs to a new class
    We simply add the label by the number of previous classes
    :param hists: histogram of predictions for each sample, size (N x C). N: #sample, C: #classes
    :param labels: array of ground truth labels, size (N, )
    :param conf_threshold: threshold for # confusions taking into account. The top num_classes are those on the diagonal
                            line of the confusion matrix. The default 5 means we consider the top 5 confusions.
    :param gt_threshold: upperbound for # occurrence of ground truth category. If the occurrence is less than this, then
                          it should be considered for relabelling
    :returns: labels_: updated array of ground truth labels, size (N, )
              equivalence: list of integers indicates the equivalence between classes, size (C_ ), C: #classes (+ new)
                equivalence[i] = j means category j is equivalent to category i. i must be one of the original classes
    """
    assert len(hists) == len(labels)

    num_classes = np.max(labels) + 1
    labels_ = np.array(labels, copy=True)
    equivalence = list(range(num_classes))
    # Create a mapping matrix that indicates whether a confusion has been assigned a new label (yes for value > 0)
    conf_map = np.zeros((num_classes, num_classes)).astype(int) - 1  # conf_mat[i][j] means misclassified as j for i

    # Compute confusion counts matrix and create new labels based on it
    conf_mat = compute_cumulative_confusion_matrix(hists, labels)  # confusion matrix
    conf_counts = np.sort(conf_mat.ravel())[::-1]  # confusion counts reversely sorted in 1d array
    conf_threshold = min(len(conf_counts), max(conf_threshold, 1))  # keep index within the range

    for i in range(len(hists)):
        h, groundtruth = hists[i], labels[i]
        majority = np.argmax(h)
        majority_count = h[majority]
        half_count = np.sum(h) / 2
        groundtruth_count = h[groundtruth]

        if majority != groundtruth and majority_count >= half_count and groundtruth_count < gt_threshold:
            if conf_mat[groundtruth, majority] > conf_counts[conf_threshold+num_classes]:  # obvious confusions
                if conf_map[groundtruth, majority] < 0:
                    conf_map[groundtruth, majority] = len(equivalence)
                    equivalence.append(groundtruth)

                labels_[i] = conf_map[groundtruth, majority]

    return labels_, equivalence


def recalibrate_categorical_ground_truth(hists, labels):
    """
    Redistributing the ground truth probability in the categorical array according to the histograms of
    outputs during training
    :param hists: histogram of predictions for each sample, size (N x C). N: #sample, C: #classes
    :param labels: labels: array of ground truth labels, size (N, )
    :return: ys: updated categorical ground truth array, size (N x C_), where C_: #classes including new subclasses
             equivalence: list of integers indicates the equivalence between classes, size (C_), C: #classes (+new)
               equivalence[i] = j means category j is equivalent to category i. i must be one of the original classes.
    """

    labels_, equivalence = relabel_training_samples(hists, labels)
    num_classes = np.max(labels_) + 1
    ys = keras.utils.to_categorical(labels_, num_classes)

    for groundtruth, label, y, hist in zip(labels, labels_, ys, hists):
        if label != groundtruth:  # recalibrate the categorical ground truth if the ground truth is updated
            groundtruth_count = hist[groundtruth]

            # Redistribute the probability to ground-truth category and the confusion category
            y[groundtruth] = max((groundtruth_count + 1.0) / np.sum(hist), 1)
            y[label] = min(1 - y[groundtruth], 0)

    return ys, equivalence


def preprocess_cifar10_data(x_train, y_train, x_test, y_test):
    """
    Preprocess the cifar10 raw data to the format which is consistent as the official cifar10_cnn.py example
    :param x_train, x_test: raw training and test images, expected to be in uint8 format
    :param y_train, y_test: raw label, in int format
    :return: (x_train, y_train), (x_test, y_test), where images are normalized to [0, 1], and labels are categorical
    """
    num_classes = max(np.max(y_train), np.max(y_test)) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if x_train.dtype == 'uint8':
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)


def evaluate_real_accuracy(pred, y, mapping=None):
    """
    Compute the real accuracy by mapping the equivalence classes to original classes
    :param pred: an array of predicted labels, (N, )
    :param y: an array of ground truth labels, (N, )
    :param mapping: an array that specifies the equivalence of sub-classes
                    mapping[i] = j means sub-class i belongs to original class j
    :return: real prediction accuracy
    """
    assert len(pred) == len(y)

    if mapping is None:
        mapping = list(range(max(y) + 1))  # use identical mapping
    else:
        assert np.max(pred) < len(mapping)

    correct_count = 0.0
    for i in range(len(pred)):
        if pred[i] == y[i] or y[i] == mapping[pred[i]]:
            correct_count += 1
    return correct_count / len(pred)


def map_probabilities(probs, mapping, num_classes, min_conf_threshold=0.0):
    """
    Map the probabilities in output vectors back to the original categories. The probability vectors are computed from
    a model that was trained with relabeled samples and the correspondence between class indices are specified in a list
    :param probs: A 2-d array of probability vectors (N x # relabeled classes), evaluated via model.predict()
    :param mapping: a list that specifies the mapping between the original class indices and the relabeled indices.
                    e.g., mapping[m] = n means that new class m is equivalent to the original category n
    :param num_classes: number of original categories
    :param min_conf_threshold: a threshold for minimum confidence level to be considered, default 0 - add prob naively
    :return: probs_ - a 2-d array of probability vectors w.r.t the original categories, size (N x num_classes)
    """
    probs_ = np.zeros((len(probs), num_classes))
    for i in range(len(probs)):
        prob = probs[i, ...]
        for j in range(len(prob)):
            if prob[j] >= min_conf_threshold:
                probs_[i, mapping[j+num_classes]] = prob[j]
    return probs_


def save_model(model, save_dir='../result/saved_models/', model_name='keras_cifar10_test_model.h5'):
    import os
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def filter_training_samples():
    # remove training samples that are consistently misclassified, already implemented in branch_net.binary_split()
    pass


def get_validation_accuracies(filepath):
    """
    Extract validation accuracies from a log file
    :param filepath: file path of the log .txt file
    :return accuracies: list of validation accuracies
    """
    accuracies = []
    lines = [line.rstrip('\n') for line in open(filepath)]
    for line in lines:
        if "val_acc" in line:
            accuracy = float(line.split(":")[-1])
            accuracies.append(accuracy)
    return accuracies


def extract_validation_accuracies(result_dir, keywords=[""], mode='any'):
    """
    Extract validation accuracies from experiment results
    :param result_dir: directory containing log files of experiments
    :param keywords: key words that we are interested in, used as filters
    :param mode: the filtering mode, match 'any' or 'all' keywords
    :return: accuracies_dict {key: filepath; value: list of accuracies}
    """
    if keywords is None:
        keywords = [""]

    accuracies_dict = {}
    filepaths = glob(result_dir + '*.txt')
    for filepath in filepaths:
        if (mode == 'any' and any(word in filepath for word in keywords)) or all(word in filepath for word in keywords):
            accuracies = get_validation_accuracies(filepath)
            accuracies_dict[filepath] = accuracies
    return accuracies_dict


def plot_accuracies():
    """
    Extract validation accuracies from log files and plot
    :return: void
    """
    result_dir = '../result/logs/'
    keywords = ["200normal"]
    accuracies_dict = extract_validation_accuracies(result_dir, keywords=keywords, mode="all")
    for experiment, accuracies in accuracies_dict.items():
        plt.plot(accuracies, label=experiment)
        print(accuracies)
    plt.legend()
    plt.show()


def plot_histogram(arr):
    """
    Plot the histogram of an array (usually the count array)
    :param arr: array that stores the counts for occurrence
    :return: void
    """
    from math import ceil
    num_bins = ceil(max(arr))
    # h = np.histogram(arr, 25)
    # print(h)
    plt.hist(arr, bins=10)
    plt.title("Histogram of # correct classification for training samples at epoch " + str(num_bins))
    plt.show()


def compute_prediction_confidence(model, x, labels):
    """
    Compute the prediction confidence
    :param model: the model
    :param x: input samples
    :param labels: ground-truth labels
    :return: conf_true, conf_false - two lists of confidence, one for correctly classified samples and the other for
             misclassified samples
    """
    assert len(x) == len(labels)
    conf_true, conf_false = [], []

    probs = model.predict(x)
    for prob, label in zip(probs, labels):
        pred = np.argmax(prob)
        conf = np.max(prob)
        if pred == label:
            conf_true.append(conf)
        else:
            conf_false.append(conf)

    return conf_true, conf_false


if __name__ == "__main__":

    plot_accuracies()

    # for epoch in range(25, 200, 25):
    #     cumulative_stats_train, cumulative_hist_train = load_pkl('online_stats_' + str(epoch) + '.pkl')
    #     plot_histogram(cumulative_stats_train)

    pass
