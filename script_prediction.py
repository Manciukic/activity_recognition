from feature_extraction import exctract_features
from read_data import import_data
import cPickle
import classifier_actions as ca

filenames = ['./data_all/Subject_10_STANDING.txt']
labels = [2]
label_array = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKDWN', 'WALKUPS']

cols = ["ACCX", "ACCY", "ACCZ"]
raw_data, labels_all, cols = import_data(filenames, labels, cols, header=0, sep="\t")

test_X, test_Y, columns = exctract_features(raw_data, labels_all, cols)

#Load previusly saved classifier
with open('./classifiers/RFclassifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

predicted_label = ca.predict(test_X, test_Y, clf, lbl_array=label_array)

print "I've predicted %s" % (label_array[predicted_label])

