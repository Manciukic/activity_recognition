from feature_extraction import exctract_features
from read_data import import_data
from tools import saveXY, build_filenames

filenames, labels = build_filenames("./data_all/Subject_",
                                    ['_LAYING.txt', '_SITTING.txt', '_STANDING.txt', '_WALKING.txt', '_WALKDWN.txt',
                                     '_WALKUPS.txt'],
                                    9)
# print filenames
# print labels
cols = ["ACCX", "ACCY", "ACCZ"]
raw_data, labels_all, cols = import_data(filenames, labels, cols, header=0, sep="\t")
# print raw_data.shape, labels_all.shape, cols
X, Y, columns = exctract_features(raw_data, labels_all, cols)

# print "[", ", ".join(columns), "]"
# print len(columns)

saveXY(X, Y, columns=columns)
