filenames = ['./data_sample/Subject_2_LAYING.txt', './data_sample/Subject_2_SITTING.txt', './data_sample/Subject_2_STANDING.txt', './data_sample/Subject_2_WALKING.txt', './data_sample/Subject_2_WALKDWN.txt', './data_sample/Subject_2_WALKUPS.txt']
# labels = ['Laying', 'Sitting', 'Standing', 'Walking', 'Walking UP', 'Walking DOWN'];
labels=[1,2,3,4,5,6,7]
cols=["ACCX", "ACCY", "ACCZ"]
raw_data, labels_all, cols = import_data(filenames, labels, cols, header=0, sep="\t")
X, Y, columns = exctract_features(raw_data, labels_all, cols)

print "X", type(X), X.shape
print "Y", type(Y), Y.shape
print "columns", type(columns), columns

