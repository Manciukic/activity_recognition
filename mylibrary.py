from __future__ import division
import numpy as np
import matplotlib as plt

def import_data (sources, labels, cols, header=1, sep=";", fsamp=50):
    '''
    Loads data for source and returns data read
    '''
    data_all=np.ndarray(shape=(0, len(cols)+1))
    labels_all=np.ndarray(shape=(0,1))
    # print "data_all.shape", data_all.shape
    # print "labels_all.shape", labels_all.shape

    start_time=0
    for i,name in enumerate(sources):
        #Reads the "name" file

        data=load_file(name, header=header, sep=sep)
        rows=data.shape[0]
        time = np.arange(0, (rows)*fsamp, fsamp) + start_time
        start_time=time[-1]+fsamp
        data=np.column_stack((time, data))
        label=np.zeros(rows) + int(labels[i])
        label.shape=(label.shape[0], 1)
    	data_all=np.row_stack((data_all, data))
    	labels_all=np.row_stack((labels_all, label))
        # print "data_all.shape", data_all.shape
        # print "labels_all.shape", labels_all.shape
    cols=["TIME"]+cols
    return data_all, labels_all, cols

def load_file(filename, header=1, sep=";"):
    '''
    Load data from file
    :param filename: name of the file where data is stored
    :return: data as np.array
    '''
    data = np.genfromtxt(filename, delimiter=sep, skip_header=header)
    return data

def extract_features_acc(data_acc, t, fsamp, col_acc, windows):
    '''
    Extract features for acceleration
    :param data_acc: data where to extract feats, as a np.array
    :param t: timestamp
    :param col_acc: labels
    :param WINLEN: window length
    :param WINSTEP: window step
    :param fsamp: sampling rate (Hz)
    :return: feats as np 2d array and the relative labels (np 1d array)
    '''

    col_mod=['ACCMOD','ACCMODPLAN']
    col_all=np.r_[col_acc, np.array(col_mod)]

    data_acc=np.column_stack((data_acc, np.sqrt(data_acc[:, 0]**2+data_acc[:,1]**2+data_acc[:,2]**2)))
    data_acc=np.column_stack((data_acc, np.sqrt(data_acc[:, 0]**2+data_acc[:,2]**2)))
    # data_acc_more, col_all=get_differences(data_acc, col_all)
    #===================================
    samples, labels=windowing_and_extraction(data_acc, t, fsamp, windows, col_all)
    return samples, labels

def extract_features_gyr(data, t, fsamp, col_gyr, windows):
    '''
    Extract features for gyroscope
    :param data: data where to extract feats, as a np.array
    :param t: timestamp
    :param col_gyr: labels
    :param WINLEN: window length
    :param WINSTEP: window step
    :param fsamp: sampling rate (Hz)
    :return: feats as np 2d array and the relative labels (np 1d array)
    '''
    # data_more, col_gyr=get_differences(data, col_gyr)
    #===================================
    samples, cols=windowing_and_extraction(data, t, fsamp,windows, col_gyr)
    return samples, cols

def extract_features_mag(data, t, fsamp, col_mag, windows):
    '''
    Extract features for magnetometer
    :param data: data where to extract feats, as a np.array
    :param t: timestamp
    :param col_mag: labels
    :param WINLEN: window length
    :param WINSTEP: window step
    :param fsamp: sampling rate (Hz)
    :return: feats as np array and the relative labels
    '''
    # data_more, col_mag=get_differences(data, col_mag)
    #===================================
    samples, cols=windowing_and_extraction(data, t, fsamp, windows, col_mag)
    return samples, cols

def windowing_and_extraction(data, t, fsamp, windows, col):
    samples = []
    columns=np.array([])

    bands=np.linspace(0.1,25,11)
    # ciclo sulla sessione - finestratura
    for t_start, t_end in windows:
        feat=np.array([])
        columns=np.array([])
        mask=(t>=t_start)&(t<t_end)

        #windowing
        data_win = data[mask,:]
        fmean=np.mean(data_win, axis=0)

        #max - mean
        feat=np.hstack([feat, np.max(data_win, axis=0)-fmean])
        columns=np.hstack([columns, concat_string(col, "_-_max_mean")])
        #print feat.shape, columns.shape
        #min - mean
        feat=np.hstack([feat, np.min(data_win, axis=0)-fmean])
        columns=np.hstack([columns,concat_string(col, '_-_min_mean')])
        #print feat.shape, columns.shape

        #max - min
        feat=np.hstack([feat, np.max(data_win, axis=0)-np.min(data_win, axis=0)])
        columns=np.hstack([columns, concat_string(col, '_-_max_min')])
        #print feat.shape, columns.shape

        #std
        feat=np.hstack([feat, np.std(data_win, axis=0)])
        columns=np.hstack([columns, concat_string(col, '_-_sd')])
        #print feat.shape, columns.shape

        #integral
        feat=np.hstack([feat, np.sum(data_win-fmean.reshape(1,fmean.shape[0]), axis=0)])
        columns=np.hstack([columns, concat_string(col, '_-_integral')])
        #print feat.shape, columns.shape

        # #mean tipo diff
        # feat=np.hstack([feat, np.mean(data_win[:,-ndiff:], axis=0)])
        # columns=np.hstack([columns, concat_string(col[-ndiff:], '_-_mean_diff')])
        # #print feat.shape, columns.shape
        #
        # #mean(abs) tipo diff
        # feat=np.hstack([feat, np.mean(np.abs(data_win[:,-ndiff:]), axis=0)])
        # columns=np.hstack([columns, concat_string(col[-ndiff:], '_-_mean_abs')])
        # #print feat.shape, columns.shape

        # FD features
        for ind_col in range(data_win.shape[1]):
            curr_col=data_win[:,ind_col]
            prefix=col[ind_col]
            curr_col_array=np.array(curr_col)
            psd=abs(np.fft.fft(curr_col_array))

            length=psd.shape[0]
            # print fsamp, length

            if length!=0 :
                fqs=np.arange(0, fsamp, fsamp/length)

                for j in range(0, len(bands)-1):
                    pw,fmx = power_fmax(psd, fqs, bands[j], bands[j+1])
                    feat=np.hstack([feat, pw, fmx])
                    columns=np.hstack([columns, prefix+'_-_power_'+str(bands[j])+'-'+str(bands[j+1]), prefix+'_-_fmax_'+str(bands[j])+'-'+str(bands[j+1])])
                    #print feat.shape, columns.shape
            else :
                print "LENGTH = 0"
                for j in range(0, len(bands) - 1):
                    pw, fmx = [0,0]
                    feat = np.hstack([feat, pw, fmx])
                    columns = np.hstack([columns, prefix + '_-_power_' + str(bands[j]) + '-' + str(bands[j + 1]),
                                         prefix + '_-_fmax_' + str(bands[j]) + '-' + str(bands[j + 1])])
                    # print feat.shape, columns.shape
        samples.append(feat)
    samples_array=np.array(samples)
    return samples_array, columns

def concat_string(array, str):
    result=[]
    for element in array:
        result.append(element+str)
    return np.array(result)

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

def exctract_features(data, labels, cols, WINLEN=2000, WINSTEP=1500, fsamp=50):
    '''
    :param data: the data
    :param labels: the labels
    :param cols: the columns in input
    :param WINLEN: window's length
    :param WINSTEP: window's step
    :return: X, Y and columns
    '''
    col_acc=["ACCX", "ACCY", "ACCZ"]
    col_gyr=["GYRX", "GYRY", "GYRZ"]
    col_mag=["MAGX", "MAGY", "MAGZ"]

    time=selectCol(data, cols, ["TIME"])

    windows, winlabs = get_windows_no_mix(time, labels, WINLEN, WINSTEP)

    try:
        data_acc=selectCol(data, cols, col_acc)
        feat_acc, fcols_acc = extract_features_acc(data_acc, time, fsamp, col_acc, windows)
    except IndexError as e:
        print "NO ACC:", e.message
        feat_acc = np.ndarray(shape=(len(windows), 0))
        fcols_acc = np.array([])

    try:
        data_gyr=selectCol(data, cols, col_gyr)
        feat_gyr, fcols_gyr = extract_features_gyr(data_gyr, time, fsamp, col_gyr, windows)
    except IndexError as e:
        print "NO GYR:", e.message
        feat_gyr = np.ndarray(shape=(len(windows), 0))
        fcols_gyr = np.array([])

    try:
        data_mag=selectCol(data, cols, col_mag)
        feat_mag, fcols_mag = extract_features_mag(data_mag, time, fsamp, col_mag, windows)
    except IndexError as e:
        print "NO MAG:", e.message
        feat_mag = np.ndarray(shape=(len(windows), 0))
        fcols_mag = np.array([])

    X = np.column_stack((feat_acc, feat_gyr, feat_mag))
    print feat_acc.shape, feat_gyr.shape, feat_mag.shape
    columns=np.r_[fcols_acc, fcols_gyr, fcols_mag]
    Y = winlabs

    return X, Y, columns

def power_fmax(spec,freq,fmin,fmax):
    #returns power in band
    psd_band=spec[np.where((freq > fmin) & (freq<=fmax))]
    # print len(psd_band)
    freq_band=freq[np.where((freq > fmin) & (freq<=fmax))]
    # print len(psd_band)
    if len(psd_band)!=0:
        powerinband = np.sum(psd_band)/len(psd_band)
        fmax=freq_band[np.argmax(psd_band)]
    else:
        powerinband=0
        fmax=0
    return powerinband, fmax

def selectCol(vect, head, cols):
    '''
    Select the cols columns from vector, given its header
    :param vect: the array to slice
    :param head: the header of the array (either as np.ndarray or list)
    :param cols: the columns to select (either as np.ndarray, list or str)
    :return: the slice of the array
    '''
    if type(head) is list:
        head=np.array(head)
    elif type(head) is not np.ndarray:
        raise ValueError("head is neither a np.ndarray or a list")

    # for i in range(len(head)):
    #     head[i]=head[i].upper()
    #
    # for i in range(len(cols)):
    #     cols[i]=cols[i].upper()

    result=np.array([])
    for col in cols:
        mask=np.zeros(len(head), dtype=bool)
        mask = (head==col)
        if result.shape[0]!=0:
            result=np.column_stack((result, vect[:,mask]))
        else:
            result=vect[:,mask]

    if result.shape[1]==1 :
        result=result.flatten()
    elif result.shape[1]==0:
        raise IndexError("No column named "+", ".join(cols))

    return result

def save(X, Y):
    np.save("X", X)
    np.save("Y", Y)

filenames = ['./data_sample/Subject_2_LAYING.txt', './data_sample/Subject_2_SITTING.txt', './data_sample/Subject_2_STANDING.txt', './data_sample/Subject_2_WALKING.txt', './data_sample/Subject_2_WALKDWN.txt', './data_sample/Subject_2_WALKUPS.txt']
# labels = ['Laying', 'Sitting', 'Standing', 'Walking', 'Walking UP', 'Walking DOWN'];
labels=[1,2,3,4,5,6,7]
cols=["ACCX", "ACCY", "ACCZ"]
raw_data, labels_all, cols = import_data(filenames, labels, cols, header=0, sep="\t")
X, Y, columns = exctract_features(raw_data, labels_all, cols)

print "X", type(X), X.shape
print "Y", type(Y), Y.shape
print "columns", type(columns), columns

