from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import numpy as np

print(__doc__)
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
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
# explained_variance = svd.explained_variance_ratio_.sum()

# ward = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
#                             connectivity=None,
#                             linkage='ward', memory=None, n_clusters=2,
#                             pooling_func='deprecated')
# ward = ward.fit(X)
# labels = ward.labels_
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))

for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    # t0 = time()
    clustering = clustering.fit(X)
    labels = clustering.labels_
    # labels = ward.labels_
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels, average_method='arithmetic'))
    # print("%s :\t%.2fs" % (linkage, time() - t0))

    # plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)

