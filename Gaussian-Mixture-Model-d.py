print(__doc__)

import time as time

import numpy as np
from sklearn.mixture import GaussianMixture
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

# #######################ward######################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 27  # number of regions
gaussian = GaussianMixture(n_components=200)
gaussian = gaussian.fit(X)
labels = gaussian.predict(X)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))