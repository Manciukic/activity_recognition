import numpy as np

from plotter import boxplot, plot_file_acc
from tools import loadXY

plot_file_acc("./data_sample/Subject_2_SITTING.txt")
plot_file_acc("./data_sample/Subject_2_WALKING.txt")

data, labels = loadXY()
columns = np.load("col.npy")

boxplot(data, columns, labels, 6, plotAll=False,
        plotCols=columns[:24],
        strlabels=['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKDWN', 'WALKUPS'])

