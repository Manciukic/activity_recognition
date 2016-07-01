import numpy as np

def concat_string(array, str):
    result=[]
    for element in array:
        result.append(element+str)
    return np.array(result)

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