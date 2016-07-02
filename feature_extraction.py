from __future__ import division

import numpy as np

from tools import concat_string, selectCol, power_fmax
from windowing import get_windows_no_mix


def extract_features_acc(data_acc, t, fsamp, col_acc, windows):
    """
    Extract features for acceleration
    :param data_acc: data where to extract feats, as a np.array
    :param t: timestamp
    :param fsamp: sampling rate (Hz)
    :param col_acc: labels
    :param windows: window array as [(tstart, tend),...]
    :return: feats as np 2d array and the relative labels (np 1d array)
    """

    col_mod = ['ACCMOD']
    col_all = np.r_[col_acc, np.array(col_mod)]

    data_acc = np.column_stack((data_acc, np.sqrt(data_acc[:, 0] ** 2 + data_acc[:, 1] ** 2 + data_acc[:, 2] ** 2)))
    # data_acc=np.column_stack((data_acc, np.sqrt(data_acc[:, 0]**2+data_acc[:,2]**2)))
    # data_acc_more, col_all=get_differences(data_acc, col_all)
    # ===================================
    samples, labels = get_features_from_windows(data_acc, t, fsamp, windows, col_all)
    return samples, labels


def extract_features_gyr(data, t, fsamp, col_gyr, windows):
    """
    Extract features for gyroscope
    :param data_acc: data where to extract feats, as a np.array
    :param t: timestamp
    :param fsamp: sampling rate (Hz)
    :param col_acc: labels
    :param windows: window array as [(tstart, tend),...]
    :return: feats as np 2d array and the relative labels (np 1d array)
    """
    # data_more, col_gyr=get_differences(data, col_gyr)
    # ===================================
    samples, cols = get_features_from_windows(data, t, fsamp, windows, col_gyr)
    return samples, cols


def extract_features_mag(data, t, fsamp, col_mag, windows):
    """
    Extract features for magnetometer
    :param data_acc: data where to extract feats, as a np.array
    :param t: timestamp
    :param fsamp: sampling rate (Hz)
    :param col_acc: labels
    :param windows: window array as [(tstart, tend),...]
    :return: feats as np 2d array and the relative labels (np 1d array)
    """
    # data_more, col_mag=get_differences(data, col_mag)
    # ===================================
    samples, cols = get_features_from_windows(data, t, fsamp, windows, col_mag)
    return samples, cols


def get_features_from_windows(data, t, fsamp, windows, col):
    """
    Retrieves the features for each window
    :param data: the raw data array
    :param t: time array
    :param fsamp: sampling rate (Hz)
    :param windows: window array as [(tstart, tend),...]
    :param col: the column names
    :return: the feature array anc the column array for the features
    """
    samples = []
    columns = np.array([])

    bands = np.linspace(0.1, 25, 11)
    # ciclo sulla sessione - finestratura
    for t_start, t_end in windows:
        feat = np.array([])
        columns = np.array([])
        mask = (t >= t_start) & (t < t_end)

        # windowing
        data_win = data[mask, :]
        fmean = np.mean(data_win, axis=0)

        # Mean
        feat = np.hstack([feat, fmean])
        columns = np.hstack([columns, concat_string(col, "_mean")])

        # Max
        feat = np.hstack([feat, np.max(data_win, axis=0)])
        columns = np.hstack([columns, concat_string(col, "_max")])

        # Min
        feat = np.hstack([feat, np.min(data_win, axis=0)])
        columns = np.hstack([columns, concat_string(col, "_min")])

        # #max - mean
        # feat=np.hstack([feat, np.max(data_win, axis=0)-fmean])
        # columns=np.hstack([columns, concat_string(col, "_-_max_mean")])
        # print feat.shape, columns.shape
        # #min - mean
        # feat=np.hstack([feat, np.min(data_win, axis=0)-fmean])
        # columns=np.hstack([columns,concat_string(col, '_-_min_mean')])
        # #print feat.shape, columns.shape

        # max - min = range
        feat = np.hstack([feat, np.max(data_win, axis=0) - np.min(data_win, axis=0)])
        columns = np.hstack([columns, concat_string(col, '_range')])
        # print feat.shape, columns.shape

        # std
        feat = np.hstack([feat, np.std(data_win, axis=0)])
        columns = np.hstack([columns, concat_string(col, '_sd')])
        # print feat.shape, columns.shape

        # integral
        feat = np.hstack([feat, integrate(data_win, t_start, t_end)])
        columns = np.hstack([columns, concat_string(col, '_integral')])
        # print feat.shape, columns.shape

        # #mean tipo diff
        # feat=np.hstack([feat, np.mean(data_win[:,-ndiff:], axis=0)])
        # columns=np.hstack([columns, concat_string(col[-ndiff:], '_-_mean_diff')])
        # #print feat.shape, columns.shape
        #
        # #mean(abs) tipo diff
        # feat=np.hstack([feat, np.mean(np.abs(data_win[:,-ndiff:]), axis=0)])
        # columns=np.hstack([columns, concat_string(col[-ndiff:], '_-_mean_abs')])
        # #print feat.shape, columns.shape

        # FD features (Fourier transformation)
        for ind_col in range(data_win.shape[1]):
            curr_col = data_win[:, ind_col]
            prefix = col[ind_col]
            curr_col_array = np.array(curr_col)
            psd = abs(np.fft.fft(curr_col_array))

            length = psd.shape[0]
            # print fsamp, length

            if length != 0:
                fqs = np.arange(0, fsamp, fsamp / length)

                # import matplotlib.pyplot as plt
                # plt.plot(fqs, psd)
                # plt.show()

                for j in range(0, len(bands) - 1):
                    pw, fmx = power_fmax(psd, fqs, bands[j], bands[j + 1])
                    feat = np.hstack([feat, pw, fmx])
                    columns = np.hstack([columns, prefix + '_power_' + str(bands[j]) + '-' + str(bands[j + 1]),
                                         prefix + '_fmax_' + str(bands[j]) + '-' + str(bands[j + 1])])
                    # print feat.shape, columns.shape
            else:
                print "LENGTH = 0"
                for j in range(0, len(bands) - 1):
                    pw, fmx = [0, 0]
                    feat = np.hstack([feat, pw, fmx])
                    columns = np.hstack([columns, prefix + '_power_' + str(bands[j]) + '-' + str(bands[j + 1]),
                                         prefix + '_fmax_' + str(bands[j]) + '-' + str(bands[j + 1])])
                    # print feat.shape, columns.shape
        samples.append(feat)
    samples_array = np.array(samples)
    return samples_array, columns


def exctract_features(data, labels, cols, WINLEN=2, WINSTEP=1.5, fsamp=10):
    """
    Extract features from the data given
    :param data: the data
    :param labels: the labels
    :param cols: the columns in input
    :param WINLEN: window's length
    :param WINSTEP: window's step
    :param fsamp: sampling frequency (Hz)
    :return: X, Y and columns
    """
    col_acc = ["ACCX", "ACCY", "ACCZ"]
    col_gyr = ["GYRX", "GYRY", "GYRZ"]
    col_mag = ["MAGX", "MAGY", "MAGZ"]

    time = selectCol(data, cols, ["TIME"])

    windows, winlabs = get_windows_no_mix(time, labels, WINLEN, WINSTEP)

    try:
        data_acc = selectCol(data, cols, col_acc)
        feat_acc, fcols_acc = extract_features_acc(data_acc, time, fsamp, col_acc, windows)
    except IndexError as e:
        print "NO ACC:", e.message
        feat_acc = np.ndarray(shape=(len(windows), 0))
        fcols_acc = np.array([])

    try:
        data_gyr = selectCol(data, cols, col_gyr)
        feat_gyr, fcols_gyr = extract_features_gyr(data_gyr, time, fsamp, col_gyr, windows)
    except IndexError as e:
        print "NO GYR:", e.message
        feat_gyr = np.ndarray(shape=(len(windows), 0))
        fcols_gyr = np.array([])

    try:
        data_mag = selectCol(data, cols, col_mag)
        feat_mag, fcols_mag = extract_features_mag(data_mag, time, fsamp, col_mag, windows)
    except IndexError as e:
        print "NO MAG:", e.message
        feat_mag = np.ndarray(shape=(len(windows), 0))
        fcols_mag = np.array([])

    X = np.column_stack((feat_acc, feat_gyr, feat_mag))
    # print feat_acc.shape, feat_gyr.shape, feat_mag.shape
    columns = np.r_[fcols_acc, fcols_gyr, fcols_mag]
    Y = winlabs

    return X, Y, columns


def integrate (data, tstart, tend):
    """
    Calculates the integral of data, an array of values of the window [tstart, tend], using the trapezes rule
    :param data: the data to integrate
    :param tstart: start time of the window
    :param tend: end time of the window
    :return: the integral calculated with trapezes rule
    """
    n = data.shape[0]
    discrete_sum = np.sum(data[1:-1], axis=0)   # discrete sum of f(a+k*delta_x), excluding f(a) and f(b)
    integral = (tend - tstart) / n * ((data[0] + data[-1]) / 2 + discrete_sum)
    return integral
