import numpy as np


def get_windows_full_label(time, labels):
    """
    The entire label length is the length of the window
    :param time: time array
    :param labels: labels array
    :return: window array as [(tstart, tend),...] and label of each window
    """
    wl = [[0, 0]]
    rl = [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            wl[-1][1] = time[i]
            wl.append([time[i], 0])
            rl.append(labels[i])
    wl[-1][1] = time[-1]
    return wl, np.array(rl)


def get_windows_no_mix(time, labels, WINLEN, WINSTEP):
    """
    calculates the windows, restarting from beginning for every new label
    :param time: time array
    :param labels: labels array
    :param WINLEN: window length
    :param WINSTEP: window step
    :return: window array as [(tstart, tend),...] and label of each window
    """
    windows = []
    labs = np.array([])
    parts, mini_labels = get_windows_full_label(time, labels)
    for start, end in parts:
        t_partial = time[(time >= start) & (time < end)]
        partial = labels[(time >= start) & (time < end)]
        starts = np.arange(t_partial[0], t_partial[-1] - WINLEN, WINSTEP)
        ends = starts + WINLEN
        for i in range(len(starts)):
            windows.append([starts[i], ends[i]])
            labs = np.r_[labs, partial[0]]
    return windows, labs
