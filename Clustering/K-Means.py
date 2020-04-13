import numpy as np
import random
from timeit import default_timer as time
# This script reads the points in CSV format
import csv
timestarter= time()


# Set up input and output variables for the script
with open('set1.csv') as csvfile: #selectring the exercise
    readCSV = csv.reader(csvfile, delimiter=',')
    # Set up CSV reader and process the header
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)  # rewind
    incsv = csv.reader(csvfile)
    if has_header:
        next(incsv)  # skip header row
    column = 1
    datatype = float
    data = (datatype(row[column]) for row in incsv)

    # Make empty lists
    z = []
    y = []
    pair = []

    # Loop through the lines in the file and get each coordinate
    for row in readCSV:
        z = float(row[0])
        y = float(row[1])
        pair.append((z, y))



def clusterdata(X, newarr):
    clusters = {}
    for x in X:
        newarrmin = min([(i[0], np.linalg.norm(x - newarr[i[0]])) \
         for i in enumerate(newarr)], key=lambda t: t[1])[0]        # start an array which has the smallets value
        # print [(i[0], np.linalg.norm(x-newarr[i[0]])) for i in enumerate(mu)]
        # print newarrmin
        try:
            clusters[newarrmin].append(x)
        except KeyError:
            clusters[newarrmin] = [x]
    return clusters


def reevaluate_centers(newarr, clusters):
    newarr2 = []
    keys = sorted(clusters.keys())
    for k in keys:
        newarr2.append(np.mean(clusters[k], axis=0))
    return newarr2


def has_converged(newarr, oldarr):
    return (set([tuple(a) for a in newarr]) == set([tuple(a) for a in oldarr]))


def findcentroids(X, K):
    # Initialize to K random centers
    oldarr = random.sample(X, K)
    newarr = random.sample(X, K)
    while not has_converged(newarr, oldarr):
        oldarr = newarr     # Assign all points in X to clusters
        clusters = clusterdata(X, newarr) # Reevaluate centers
        newarr = reevaluate_centers(oldarr, clusters)
        return (newarr, clusters)


def init_board(N):
    X = np.array([(pair[i][0], pair[i][1]) for i in range(N)])
    return X


X = init_board(len(pair))

newarr, clusters = findcentroids(X,4) # introduce the K value as (X, K)
endtimer=time()
print newarr # print to checking the centroids
print "time in seconds= "+ str(endtimer-timestarter)

import matplotlib.pyplot as plt

xValues = []
yValues = []

for x in clusters:
    nextX = []
    nextY = []
    for point in clusters[x]:
        nextX.append(point[0])
        nextY.append(point[1])
    xValues.append(nextX)
    yValues.append(nextY)

colour = ["blue", "red", "black", "yellow"]

for i in range(0,len(newarr)):
    plt.scatter(xValues[i], yValues[i], s=75, c=colour[i]) #xValues and yValues must be the same length
    plt.scatter(newarr[i][0], newarr[i][1], s=150, c=colour[i])
plt.show()
