print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
X = scale(digits.data)

n_samples, n_features = X.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target

sample_size = 300

# # #############################################################################
# # Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
#                             random_state=0)

# #############################################################################
# Compute Affinity Propagation
aff = AffinityPropagation(preference=-30)
aff = aff.fit(X)
# cluster_centers_indices = af.cluster_centers_indices_
labels = aff.labels_

# n_clusters_ = len(cluster_centers_indices)
#
# print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))


# # #############################################################################
# # Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# plt.close('all')
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()