from sklearn.ensemble import RandomForestClassifier
from tools import loadXY
import cPickle

X, Y = loadXY()

clf = RandomForestClassifier()
clf = clf.fit(X, Y)

#Save classifier to file
with open('./classifiers/RFclassifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)
