from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np

print(__doc__)
categories = [
    'alt.atheism',
    'talk.religion.misc',
]

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
labels_true = dataset.target
true_k = np.unique(labels_true).shape[0]
# t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)
svd = TruncatedSVD(2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

# Compute DBSCAN
dbscan = DBSCAN(eps=6.5, min_samples=3)
dbscan = dbscan.fit(X)
# core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
# core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))
