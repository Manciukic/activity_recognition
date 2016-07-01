import matplotlib.pyplot as plt
import numpy as np
from read_data import load_file
from tools import selectCol


def boxplot(data, columns, labels, n_labels, plotAll=True, plotCols=[], strlabels=[]):
    """
    Makes a boxplot for the specified columns
    :param data: the data matrix
    :param columns: the total columns
    :param labels: the labels array
    :param n_labels: the number of different labels
    :param plotAll: wanna plot all columns?
    :param plotCols: columns to plot
    :param strlabels: the name of the labels to print
    :return:
    """
    if plotAll:
        plotCols = columns

    selected_columns = selectCol(data, columns, plotCols)

    min_len = get_min_length(labels, n_labels)

    for c in range(selected_columns.shape[1]):
        data_column = selected_columns[:, c]
        data_to_plot = np.ndarray(shape=(min_len, 0))
        for i in range(n_labels):
            mask = labels == i
            labeled_data = data_column[mask]
            if labeled_data.shape[0] > min_len:
                labeled_data = labeled_data[:min_len]
            data_to_plot = np.column_stack((data_to_plot, labeled_data))

        plt.figure()
        plt.boxplot(data_to_plot, labels=strlabels)
        plt.title(plotCols[c])
    plt.show()


def get_min_length(labels, n):
    """
    Returns the min length of the labels
    :param labels: the labels array
    :param n: the number of different labels
    :return: the min length
    """
    min = 50000000000  # inf
    for i in range(n):
        mask = labels == i
        length = labels[mask].shape[0]
        if length < min:
            min = length
    return min

def plot_file_acc(filename, header=0, sep="\t", srate=0.1):
    data = load_file(filename, header=header, sep=sep)
    time = np.linspace(0, 0 + (data.shape[0]) * srate, num=data.shape[0], endpoint=False)
    plt.plot(time, data)
    plt.show()
