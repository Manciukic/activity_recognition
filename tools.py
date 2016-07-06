import numpy as np


def concat_string(array, str):
    """
    Concatenates every string in array with str
    :param array: string array
    :param str: string
    :return: concatenated string array
    """
    result = []
    for element in array:
        result.append(element + str)
    return np.array(result)


def power_fmax(spec, freq, fmin, fmax):
    """
    Returns power in band and maximum frequency in the selected interval
    :param spec: spectrum of frequencies
    :param freq: frequencies array
    :param fmin: lower bound
    :param fmax: upper bound
    :return: power in band and maximum frquency
    """
    # returns power in band
    psd_band = spec[np.where((freq > fmin) & (freq <= fmax))]
    # print len(psd_band)
    freq_band = freq[np.where((freq > fmin) & (freq <= fmax))]
    # print len(psd_band)
    if len(psd_band) != 0:
        powerinband = np.sum(psd_band) / len(psd_band)
        fmax = freq_band[np.argmax(psd_band)]
    else:
        powerinband = 0
        fmax = 0
    return powerinband, fmax


def selectCol(vect, head, cols):
    """
    Select the cols columns from vector, given its header
    :param vect: the array to slice
    :param head: the header of the array (either as np.ndarray or list)
    :param cols: the columns to select (either as np.ndarray, list or str)
    :return: the slice of the array
    """
    if type(head) is list:
        head = np.array(head)
    elif type(head) is not np.ndarray:
        raise ValueError("head is neither a np.ndarray or a list")

    # for i in range(len(head)):
    #     head[i]=head[i].upper()
    #
    # for i in range(len(cols)):
    #     cols[i]=cols[i].upper()

    result = np.array([])
    for col in cols:
        mask = np.zeros(len(head), dtype=bool)
        mask = (head == col)
        if result.shape[0] != 0:
            result = np.column_stack((result, vect[:, mask]))
        else:
            result = vect[:, mask]

    if result.shape[1] == 1:
        result = result.flatten()
    elif result.shape[1] == 0:
        raise IndexError("No column named " + ", ".join(cols))

    return result


def saveXY(X, Y, columns=[]):
    """
    Saves X and Y arrays
    :param X: Feature array as np.ndarray
    :param Y: Labels array as np.array
    :return: null
    """
    np.save("X", X)
    np.save("Y", Y)
    if len(columns) != 0:
        np.save("col", np.array(columns))


def loadXY():
    """
    Loads X and Y array from working directory
    :return: X and Y arrays
    """
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    return X, Y


def build_filenames(prefix, suffixes, n):
    """
    Build the list of file names given the number
    :param prefix: the prefix of the filename
    :param suffixes: the list of all the suffixes of the file names
    :param n: the number of subjects
    :return: the list of file names and the labels
    """
    filenames = []
    labels = []
    for l, suffix in enumerate(suffixes):
        for i in range(1, 1 + n):
            filenames.append(prefix + str(n) + suffix)
            labels.append(l)
    return filenames, labels

def module(data):
    """
    Calculates vector module
    :param data: np.ndarray (shape = n,3), rows = values, columns= x, y, z
    :return: the vector module column as a np.array 1d
    """
    return np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)