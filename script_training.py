from sklearn.ensemble import RandomForestClassifier

import training as tr
from tools import loadXY

X, Y = loadXY()

# Test values for our classificator
[selX, selY] = tr.pick_random_values_stratified(X, Y)

clf = RandomForestClassifier()

tr.predict_evaluate(X, Y, clf)
