import copy
import numpy as np

X = np.genfromtxt('Data.tsv', delimiter="\t")
# num of clusters
k = 3
# initial centroids as detailed in assignment
cents = [[1.03800476, 0.09821729, 1.0469454, 1.58046376],
         [0.18982966, -1.97355361, 0.70592084, 0.3957741],
         [1.2803405, 0.09821729, 0.76275827, 1.44883158]]
numDatum = X.shape[0]
numFeatures = X.shape[1]


# initialize model
class kMeans(object):
    def __init__(self):
        self = self
        self.k = k
        # keep track of centroids and old ones
        self.centroids = np.array(cents)
        self.oldCentroids = np.array(cents)
        # initialize error as inf and distances and assignments as zeros
        self.dists = np.zeros((numFeatures, k))
        self.error = float("inf")
        self.assignments = np.zeros(numDatum)

    def fit(self, X):
        # run til convergence
        while self.error > 0:
            i = 0
            # choose centroid closest to features of data point
            for feat in X:
                distances = [np.linalg.norm(feat - x) for x in self.centroids]
                decision = distances.index(min(distances))
                self.assignments[i] = decision
                i = i + 1
            # deepcopy to avoid chained changes/dependencies, my code significantly underperformed without this, found fix on stackoverflow
            self.oldCentroids = copy.deepcopy(self.centroids)
            # update centroids
            for i in range(len(self.centroids)):
                self.centroids[i] = np.mean(X[self.assignments == i], axis=0)
            # distance between/norm of new and old centroids
            self.error = np.linalg.norm(self.centroids - self.oldCentroids)
        # output centroids to use as starter for GMM
        np.savetxt('kmeans_cents.tsv', X=self.centroids, delimiter="\t")
        # output assignments
        np.savetxt('kmeans_output.tsv', X=self.assignments, delimiter="\t")


predict = kMeans()
predict.fit(X)
