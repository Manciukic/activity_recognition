from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import training as tr
from tools import loadXY

X, Y = loadXY()

# Test values for our classificator
# [selX, selY] = tr.pick_random_values_stratified(X, Y)

clf = RandomForestClassifier()
print "RandomForest",
tr.cross_validation(X, Y, clf)

clf = LinearSVC()
print "Linear SVC",
tr.cross_validation(X, Y, clf)

clf = AdaBoostClassifier()
print "Adaboost",
tr.cross_validation(X, Y, clf)

clf = KNeighborsClassifier()
print "NearestNeighbors",
tr.cross_validation(X, Y, clf)

clf = DecisionTreeClassifier()
print "DecisionTree",
tr.cross_validation(X, Y, clf)

clf = SVC()
print "SVM RBF",
tr.cross_validation(X, Y, clf)
