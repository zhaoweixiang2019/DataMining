# 								实验报告

## 实验名称

实验一 基于sklearn的聚类实验

## 实验要求

测试sklearn中多种聚类算法在两种数据集上的聚类效果。

## 聚类算法

K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical、Agglomerative clustering、DBSCAN、Gaussian mixtures

## 数据集

sklearn.datasets.load_di gits、 sklearn.datasets.fetch_2 0newsgroups

## 评估函数

Normalized Mutual Information (NMI)、Homogeneity: each cluster contains only members of a single class 、 Completeness: all members of a given class are assigned to the same cluster 

## 实验环境

硬件环境：Intel(R) Core(TM) i5-3470 CPU @ 3.20GHz

软件环境：windows 10、pycharm

编程语言：python 3.7

## 实验过程

首先导入各种所需要用到的包。

然后从load_digits中得到data和digits。

然后从digits和data中得到需要的lebels_true和lesbels。

最后使用各种聚类方法在数据集上进行实验。

**K-Means**

​	k-means算法实际上就是通过计算不同样本间的距离来判断他们的相近关系的，相近的就会放到同一个类别中去。

 	1.首先选择一个k值，也就是希望把数据分成多少类，这里k值的选择对结果的影响很大，选择方法有两种一种是elbow method，简单的说就是根据聚类的结果和k的函数关系判断k为多少的时候效果最好。另一种则是根据具体的需求确定，比如说进行衬衫尺寸的聚类你可能就会考虑分成三类（L,M,S）等

​	 2.然后选择最初的聚类点（或者叫质心），这里的选择一般是随机选择的，代码中的是在数据范围内随机选择，另一种是随机选择数据中的点。这些点的选择会很大程度上影响到最终的结果，也就是说运气不好的话就到局部最小值去了。这里有两种处理方法，一种是多次取均值，另一种则是后面的改进算法（bisecting K-means）

​	 3.接下来把数据集中所有的点都计算下与这些质心的距离，把它们分到离它们质心最近的那一类中去。完成后我们则需要将每个簇算出平均值，用这个点作为新的质心。反复重复这两步，直到收敛我们就得到了最终的结果。

**Affinity Propagation**

在统计和数据挖掘里，affinity propagation(AP)是一种基于数据点之间的“信息传递”的聚类算法。与k-means等其它聚类算法不同的是，AP不需要在聚类前确定或估计类的个数。类似于k-medoids, AP需要寻找原型(exemplars), 即，代表类的输入集里的成员。AP算法广泛应用于计算机视觉和计算生物学领域。

**Mean-Shift**

1. 在未被标记的数据点中随机选择一个点作为起始中心点center；
2. 找出以center为中心半径为radius的区域中出现的所有数据点，认为这些点同属于一个聚类C。同时在该聚类中记录数据点出现的次数加1。
3. 以center为中心点，计算从center开始到集合M中每个元素的向量，将这些向量相加，得到向量shift。
4. center = center + shift。即center沿着shift的方向移动，移动距离是||shift||。
5. 重复步骤2、3、4，直到shift的很小（就是迭代到收敛），记住此时的center。注意，这个迭代过程中遇到的点都应该归类到簇C。
6. 如果收敛时当前簇C的center与其它已经存在的簇C2中心的距离小于阈值，那么把C2和C合并，数据点出现次数也对应合并。否则，把C作为新的聚类。
7. 重复1、2、3、4、5直到所有的点都被标记为已访问。
8. 分类：根据每个类，对每个点的访问频率，取访问频率最大的那个类，作为当前点集的所属类。

**Spectral Clustering**

 	1. 构建一个相似度矩阵
 	2. 构建拉普拉斯矩阵L
 	3. 归一化矩阵L
 	4. 计算矩阵L的特征值和特征向量

**Agglomerative clustering**

1. 将每一个元素单独定为一类
2. 重复：每一轮都合并指定距离(对指定距离的理解很重要)最小的类
3. 直到所有的元素都归为同一类

**DBSCAN**

​	DBScan需要二个参数： 扫描半径 (eps)和最小包含点数(minPts)。 任选一个未被访问(unvisited)的点开始，找出与其距离在eps之内(包括eps)的所有附近点。

​	 如果 附近点的数量 ≥ minPts，则当前点与其附近点形成一个簇，并且出发点被标记为已访问(visited)。 然后递归，以相同的方法处理该簇内所有未被标记为已访问(visited)的点，从而对簇进行扩展。

​	 如果 附近点的数量 < minPts，则该点暂时被标记作为噪声点。

​	 如果簇充分地被扩展，即簇内的所有点被标记为已访问，然后用同样的算法去处理未被访问的点。

**Gaussian mixtures**

​	对图像背景建立高斯模型的原理及过程：图像灰度直方图反映的是图像中某个灰度值出现的频次，也可以认为是图像灰度概率密度的估计。如果图像所包含的目标区域和背景区域相比比较大，且背景区域和目标区域在灰度上有一定的差异，那么该图像的灰度直方图呈现双峰-谷形状，其中一个峰对应于目标，另一个峰对应于背景的中心灰度。对于复杂的图像，尤其是医学图像，一般是多峰的。通过将直方图的多峰特性看作是多个高斯分布的叠加，可以解决图像的分割问题。 在智能监控系统中，对于运动目标的检测是中心内容，而在运动目标检测提取中，背景目标对于目标的识别和跟踪至关重要。而建模正是背景目标提取的一个重要环节。

## 实验结果

数据集digits简称d、数据集20newsgroup简称20

**K-Means**

数据集d：

n_digits: 10, 	 n_samples 1797, 	 n_features 64

__________________________________________________________________________________

init		          time	  inertia	homo	compl	NML
k-means++	0.23s	69432	0.602	0.650	0.625

__________________________________________________________________________________

数据集20：

Homogeneity: 0.368
Completeness: 0.400
NML: 0.384

**Affinity Propagation**

数据集d：

Homogeneity: 0.984
Completeness: 0.396
NMI: 0.564

数据集20：

Homogeneity: 1.000
Completeness: 0.094
NMI: 0.173

**Mean-Shift**

数据集d：

Homogeneity: 0.008
Completeness: 0.250
NMI: 0.016

数据集20：

Homogeneity: 0.089
Completeness: 0.074
NMI: 0.081

**Spectral Clustering**

数据集d：

Homogeneity: 0.773
Completeness: 0.878
NMI: 0.822

数据集20：

Homogeneity: 0.456
Completeness: 0.305
NMI: 0.366

**Agglomerative clustering**

数据集d：

ward:
Homogeneity: 0.758
Completeness: 0.836
NMI: 0.796
average:
Homogeneity: 0.007
Completeness: 0.238
NMI: 0.014
complete:
Homogeneity: 0.017
Completeness: 0.249
NMI: 0.032
single:
Homogeneity: 0.006
Completeness: 0.276
NMI: 0.011

数据集20：

Homogeneity: 0.476
Completeness: 0.295
NMI: 0.364
Homogeneity: 0.477
Completeness: 0.307
NMI: 0.374
Homogeneity: 0.482
Completeness: 0.303
NMI: 0.372
Homogeneity: 0.010
Completeness: 0.173
NMI: 0.019

**DBSCAN**

数据集d：

Homogeneity: 0.549
Completeness: 0.516
NMI: 0.532

数据集20：

Homogeneity: 0.000
Completeness: 1.000
NMI: 0.000

**Gaussian mixtures**

数据集d：

Homogeneity: 0.962
Completeness: 0.436
NMI: 0.600

数据集20：

Homogeneity: 0.450
Completeness: 0.095
NMI: 0.157

## 总结

刚开始接触python，所以代码写的可能不是很好，这多个文件可以合并成一个文件。
