from feature_extraction import exctract_features
from read_data import import_data
from tools import saveXY, build_filenames

# filenames, labels = build_filenames("./data_all/Subject_",
#                                     ['_LAYING.txt', '_SITTING.txt', '_STANDING.txt', '_WALKING.txt', '_WALKDWN.txt',
#                                      '_WALKUPS.txt'],
#                                     9)
filenames, labels = build_filenames("./data_collected/Subject_",
                                    ['_LAYING.txt', '_SITTING.txt', '_STANDING.txt', '_WALKING.txt', '_RUNNING.txt'],
                                    3)

# filenames = ["./data_collected/Subject_1_SITTING3.txt"]
# labels = [0]

cols = ["ACCX", "ACCY", "ACCZ", "GYRX", "GYRY", "GYRZ"]
raw_data, labels_all, cols = import_data(filenames, labels, cols, header=5, sep=",", cut_sec=5, srate=0.04)
# raw_data, labels_all, cols = import_data(filenames, labels, cols, header=0, sep="\t")
print raw_data.shape[0]*0.04

X, Y, columns = exctract_features(raw_data, labels_all, cols, fsamp=25)

saveXY(X, Y, columns=columns)
