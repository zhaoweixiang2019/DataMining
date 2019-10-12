print(__doc__)

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import SpectralClustering
from sklearn import metrics

np.random.seed(42)
digits = load_digits()
X = scale(digits.data)
n_samples, n_features = X.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300

sp = SpectralClustering(affinity="nearest_neighbors")
sp = sp.fit(X)
labels = sp.labels_

# print("number of estimated clusters : %d" % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))


