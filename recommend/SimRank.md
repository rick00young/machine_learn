# SimRank协同过滤推荐算法
[http://www.cnblogs.com/pinard/p/6362647.html](http://www.cnblogs.com/pinard/p/6362647.html)
## 1. SimRank推荐算法的图论基础
　　　　SimRank是基于图论的，如果用于推荐算法，则它假设用户和物品在空间中形成了一张图。而这张图是一个二部图。所谓二部图就是图中的节点可以分成两个子集，而图中任意一条边的两个端点分别来源于这两个子集。一个二部图的例子如下图。从图中也可以看出，二部图的子集内部没有边连接。对于我们的推荐算法中的SimRank，则二部图中的两个子集可以是用户子集和物品子集。而用户和物品之间的一些评分数据则构成了我们的二部图的边。

## 2. SimRank推荐算法思想
　　　　对于用户和物品构成的二部图，如何进行推荐呢？SimRank算法的思想是，如果两个用户相似，则与这两个用户相关联的物品也类似；如果两个物品类似，则与这两个物品相关联的用户也类似。如果回到上面的二部图，假设上面的节点代表用户子集，而下面节点代表物品子集。如果用户1和3类似，那么我们可以说和它们分别相连的物品2和4也类似。

　　　　如果我们的二部图是G(V,E)，其中V是节点集合，E是边集合。则某一个子集内两个点的相似度s(a,b)可以用和相关联的另一个子集节点之间相似度表示。即：
　　　　$$s(a,b)=\frac{C}{|I(a)||I(b)|}\sum_{i=1}^{|I(a)|}\sum_{j=1}^{|I(b)|}s(I_i(a), I_i(b))$$

其中C是一个常数，而$I(a),I(b)$分别代表和a，b相连的二部图另一个子集的节点集合。$s(I_i(a),I_i(b))$即为相连的二部图另一个子集节点之间的相似度。

## 4. SimRank++算法原理
　　　　SimRank++算法对SimRank算法主要做了两点改进。第一点是考虑了边的权值，第二点是考虑了子集节点相似度的证据。

　　　　对于第一点边的权值，上面的SimRank算法，我们对于边的归一化权重，我们是用的比较笼统的关联的边数分之一来度量，并没有考虑不同的边可能有不同的权重度量，而SimRank++算法则在构建转移矩阵W时会考虑不同的边的不同权重值这个因素。

　　　　对于第二点的节点相似度的证据。回顾回顾上面的SimRank算法，我们只要认为有边相连，则为相似。却没有考虑到如果共同相连的边越多，则意味着两个节点的相似度会越高。而SimRank++算法利用共同相连的边数作为证据，在每一轮迭代过程中，对SimRank算法计算出来的节点相似度进行修正，即乘以对应的证据值得到当前轮迭代的的最终相似度值。

## 5. SimRank系列算法的求解
　　　　由于SimRank算法涉及矩阵运算，如果用户和物品量非常大，则对应的计算量是非常大的。如果直接用我们第二节讲到了迭代方法去求解，所花的时间会很长。对于这个问题，除了传统的一些SimRank求解优化以外，常用的有两种方法来加快求解速度。

　　　　第一种是利用大数据平台并行化，即利用Hadoop的MapReduce或者Spark来将矩阵运算并行化，加速算法的求解。

　　　　第二种是利用蒙特卡罗法(Monte Carlo, MC)模拟，将两结点间 SimRank 的相似度表示为两个随机游走者分别从结点 a和 b出发到最后相遇的总时间的期望函数。用这种方法时间复杂度会大大降低，但是由于MC带有一定的随机性，因此求解得到的结果的精度可能不高。

## 6. SimRank小结
　　　　作为基于图论的推荐算法，目前SimRank算法在广告推荐投放上使用很广泛。而图论作为一种非常好的建模工具，在很多算法领域都有广泛的应用，比如我之前讲到了谱聚类算法。同时，如果你理解了SimRank，那么Google的PageRank对你来说就更容易理解了。