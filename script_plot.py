import numpy as np

from plotter import boxplot
from tools import loadXY

data, labels = loadXY()
columns = np.load("col.npy")

boxplot(data, columns, labels, 6, plotAll=False,
        plotCols=columns[:24],
        strlabels=['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKDWN', 'WALKUPS'])
