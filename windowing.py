import numpy as np

def get_windows_full_label(time, labels):
    '''
    Window = Label length
    :param labels: np.array of labels
    :return: windows as list of [start, end]
    '''
    wl= [[0,0]]
    rl= [labels[0]]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
    	    wl[-1][1]= time[i]
            wl.append([time[i],0])
            rl.append(labels[i])
    wl[-1][1]= time[-1]
    return wl, np.array(rl)

def get_windows_no_mix(time, labels, WINLEN, WINSTEP):
    '''
    calculates the windows, restarting from beginning for every new label
    return: (windows, labels). windows is a list of [start, end], label is a list of int
    labels: array of labels (int)
    '''
    windows=[]
    labs=np.array([])
    parts, mini_labels = get_windows_full_label(time, labels)
    for start, end in parts:
        t_partial=time[(time>=start) & (time<end)]
        partial=labels[(time>=start) & (time<end)]
        starts=np.arange(t_partial[0], t_partial[-1]-WINLEN, WINSTEP)
        ends=starts+WINLEN
        for i in range(len(starts)):
            windows.append([starts[i], ends[i]])
            labs=np.r_[labs, partial[0]]
    return windows, labs