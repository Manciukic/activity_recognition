import numpy as np


def import_data(sources, labels, cols, header=1, sep=";", srate=0.1):
    """
    Loads data for source and returns data read
    :param sources: the file names to read from
    :param labels: the labels of each file
    :param cols: the columns of the files (ex: ['ACCX', 'ACCY', 'ACCZ'])
    :param header: the number of lines to skip
    :param sep: the separator used in the csv file
    :param srate: the sampling rate used (in s)
    :return: the data read as a np.ndarray, the labels as a np.ndarray, the columns as a list
    """
    data_all = np.ndarray(shape=(0, len(cols) + 1))
    labels_all = np.ndarray(shape=(0, 1))
    # print "data_all.shape", data_all.shape
    # print "labels_all.shape", labels_all.shape

    start_time = 0
    for i, name in enumerate(sources):
        # Reads the "name" file

        data = load_file(name, header=header, sep=sep)
        rows = data.shape[0]
        time = np.linspace(start_time, start_time + (rows) * srate, num=rows, endpoint=False)
        start_time = time[-1] + srate
        data = np.column_stack((time, data))
        label = np.zeros(rows) + int(labels[i])
        label.shape = (label.shape[0], 1)
        data_all = np.row_stack((data_all, data))
        labels_all = np.row_stack((labels_all, label))
        # print "data_all.shape", data_all.shape
        # print "labels_all.shape", labels_all.shape
    cols = ["TIME"] + cols
    return data_all, labels_all, cols


def load_file(filename, header=1, sep=";"):
    """
    Load data from file
    :param filename: name of the file where data is stored
    :return: data as np.array
    """
    data = np.genfromtxt(filename, delimiter=sep, skip_header=header)
    return data
