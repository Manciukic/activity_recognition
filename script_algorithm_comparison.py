from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import classifier_actions as ca
from tools import loadXY

X, Y = loadXY()

# Test values for our classifier
# [selX, selY] = tr.pick_random_values_stratified(X, Y)

clf = RandomForestClassifier()
print "RandomForest",
ca.cross_validation(X, Y, clf)

clf = LinearSVC()
print "Linear SVC",
ca.cross_validation(X, Y, clf)

clf = AdaBoostClassifier()
print "Adaboost",
ca.cross_validation(X, Y, clf)

clf = KNeighborsClassifier()
print "NearestNeighbors",
ca.cross_validation(X, Y, clf)

clf = DecisionTreeClassifier()
print "DecisionTree",
ca.cross_validation(X, Y, clf)

clf = SVC()
print "SVM RBF",
ca.cross_validation(X, Y, clf)
