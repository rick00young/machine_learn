## 线性回归

### 局部加权线性回归:Locally Weighted Linear Regression , LWLR
使用核函数

高斯核:

$$\hat{w} = (X^TWX)^{-1}X^TWy$$

高斯核对的权重如下:

$$w(i,j)=exp\Biggl(\frac{|x^{(i)}-x|}{-2k^2}\Biggr)$$


缺点:

局部加权线性增加了计算量,它对每个点做预测时都必须使用整个数据集

### 岭回归: ridge regression
岭回归最先用来处理特征数多于样本数的情况,也用于在估计中加入偏差,从而得到更好的估计.通过引入$\lambda$来限制的所有w的和,通过引入该处罚项,可以减少不重要的参数,这种技术叫做缩减(shrinkage)


岭回归是矩阵 $X^TX$上加一下 $\lambda{I}$ 从而使得矩阵非厅异,进而可以求 $X^TX+\lambda{I}$ 求逆, 其中矩阵 $I$ 是一个mxm的单位矩阵,对角线上无线全为1,其他元素全为0(可以看做是一条线划分为岭).$\lambda$是用户定义的数值.回归系统计算公式变为:

$$\hat{w}=(X^TX+\lambda{I})^{-1}X^Ty$$


### lasso
在增加如下约束时,普通的最小二乘法回归会得到与岭回归一样的公式:

$$\sum_{k=1}^{n}w^2_k\le\lambda$$

使用普通的最小二乘回归当两个或更多的特征相关时,可能会得到一个很大的正系数和一个很大的负系数.使用岭回归可以避免这个问题.


lasso也对回归系数做了限定:
$$\sum_{k=1}^{n}\mid{w_k}\mid\le\lambda$$