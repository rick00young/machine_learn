## Factorization Machines


(http://blog.csdn.net/itplus/article/details/40534885)[http://blog.csdn.net/itplus/article/details/40534885]

Factorization Machines 简称:FM
它可对任意的实值向量进行预测。

其主要优点包括:

* 1).可用于高度稀疏数据场景；
* 2).具有线性的计算复杂度。

本文将对 FM 框架进行简单介绍，并对其训练算法 — 随机梯度下降（SGD）法和交替最小二乘法（ALS）法进行详细推导。