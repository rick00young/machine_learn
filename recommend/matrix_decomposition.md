# 矩阵分解在协同过滤推荐算法中的应用
[http://www.cnblogs.com/pinard/p/6351319.html](http://www.cnblogs.com/pinard/p/6351319.html)

## 2. 传统的奇异值分解SVD用于推荐
说道矩阵分解，我们首先想到的就是奇异值分解SVD。在奇异值分解(SVD)原理与在降维中的应用中，我们对SVD原理做了总结。如果大家对SVD不熟悉的话，可以翻看该文。

此时可以将这个用户物品对应的m×n矩阵MM进行SVD分解，并通过选择部分较大的一些奇异值来同时进行降维，也就是说矩阵MM此时分解为：
$$M_{mxn}=U_{mxk}Σ_{kxk}V_{kxn}^{T}$$
其中k是矩阵MM中较大的部分奇异值的个数，一般会远远的小于用户数和物品树。如果我们要预测第i个用户对第j个物品的评分mijmij,则只需要计算uTiΣvjuiTΣvj即可。通过这种方法，我们可以将评分表里面所有没有评分的位置得到一个预测评分。通过找到最高的若干个评分对应的物品推荐给用户。

可以看出这种方法简单直接，似乎很有吸引力。但是有一个很大的问题我们忽略了，就是SVD分解要求矩阵是稠密的，也就是说矩阵的所有位置不能有空白。有空白时我们的MM是没法直接去SVD分解的。大家会说，如果这个矩阵是稠密的，那不就是说我们都已经找到所有用户物品的评分了嘛，那还要SVD干嘛! 的确，这是一个问题，传统SVD采用的方法是对评分矩阵中的缺失值进行简单的补全，比如用全局平均值或者用用户物品平均值补全，得到补全后的矩阵。接着可以用SVD分解并降维。

虽然有了上面的补全策略，我们的传统SVD在推荐算法上还是较难使用。因为我们的用户数和物品一般都是超级大，随便就成千上万了。这么大一个矩阵做SVD分解是非常耗时的。那么有没有简化版的矩阵分解可以用呢？我们下面来看看实际可以用于推荐系统的矩阵分解。

## 3. FunkSVD算法用于推荐
　　　　FunkSVD是在传统SVD面临计算效率问题时提出来的，既然将一个矩阵做SVD分解成3个矩阵很耗时，同时还面临稀疏的问题，那么我们能不能避开稀疏问题，同时只分解成两个矩阵呢？也就是说，现在期望我们的矩阵MM这样进行分解：
$$M_{mxn}=P_{mxk}^TQ_{kxn}$$
　　　　我们知道SVD分解已经很成熟了，但是FunkSVD如何将矩阵MM分解为P和Q呢？这里采用了线性回归的思想。我们的目标是让用户的评分和用矩阵乘积得到的评分残差尽可能的小，也就是说，可以用均方差作为损失函数，来寻找最终的PP和Q。

　　　　对于某一个用户评分$m_{ij}$，如果用FunkSVD进行矩阵分解，则对应的表示为$q_i^Tp_i$，采用均方差做为损失函数，则我们期望$(m_{ij}-q_j^Tp_i)^2$尽可能的小，如果考虑所有的物品和样本的组合，则我们期望最小化下式：
$$\sum_{i,j}(m_{ij}-q_j^Tp_i)^2$$

　　　　只要我们能够最小化上面的式子，并求出极值所对应的$p_i,q_j$则我们最终可以得到矩阵P和Q，那么对于任意矩阵M任意一个空白评分的位置，我们可以通过$q_j^Tp_i$计算预测评分。很漂亮的方法！

　　　　当然，在实际应用中，我们为了防止过拟合，会加入一个L2的正则化项，因此正式的FunkSVD的优化目标函数J(p,q)是这样的：
$$\underbrace{arg min}_{p_i, q_j}\sum_{i,j}(m_{ij}-q_j^Tp_i)^2 + \lambda(||p_i||_2^2+||q_j||_2^2)$$
　　　　其中λ为正则化系数，需要调参。对于这个优化问题，我们一般通过梯度下降法来进行优化得到结果。

　　　　将上式分别对$p_i, q_j$,求导我们得到:

 $$\frac{\partial J}{\partial p_i}=-2(m_{ij}-q_j^Tp_i)q_j+2\lambda p_i$$
 $$\frac{\partial L}{\partial q_j}=-2(m_{ij}-q_j^Tp_i)pI+2\lambda q_j$$

　　　　则在梯度下降法迭代时，$p_i,q_j$的迭代公式为：
$$p_i=p_i+\alpha((m_{ij}-q_j^Tp_i)q_j-\lambda p_i)$$
$$q_j=q_j+\alpha((m_{ij}-q_j^Tp_i)p_i-\lambda q_j)$$
　　　　通过迭代我们最终可以得到P和Q，进而用于推荐。FunkSVD算法虽然思想很简单，但是在实际应用中效果非常好，这真是验证了大道至简。

## 5. SVD++算法用于推荐
[SVD++ Implementation in GraphX](http://www.farseer.cn/2015/08/16/svd-implementation-in-graphx/)

## 6. 矩阵分解推荐方法小结
　　　　FunkSVD将矩阵分解用于推荐方法推到了新的高度，在实际应用中使用也是非常广泛。当然矩阵分解方法也在不停的进步，目前张量分解和分解机方法是矩阵分解推荐方法今后的一个趋势。

　　　　对于矩阵分解用于推荐方法本身来说，它容易编程实现，实现复杂度低，预测效果也好，同时还能保持扩展性。这些都是它宝贵的优点。当然，矩阵分解方法有时候解释性还是没有基于概率的逻辑回归之类的推荐算法好，不过这也不影响它的流形程度。小的推荐系统用矩阵分解应该是一个不错的选择。大型的话，则矩阵分解比起现在的深度学习的一些方法不占优势。