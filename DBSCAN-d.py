print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
np.random.seed(42)
digits = load_digits()
X = scale(digits.data)
n_samples, n_features = X.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target
sample_size = 300

# Compute DBSCAN
dbscan = DBSCAN(eps=4.5, min_samples=1)
dbscan = dbscan.fit(X)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))

