## Lasso回归
Lasso回归有时也叫做线性回归的L1正则化，和Ridge回归的主要区别就是在正则化项，Ridge回归用的是L2正则化，而Lasso回归用的是L1正则化。Lasso回归的损失函数表达式如下:
$$J(\theta)=\frac{1}{2n}(X\theta-Y)^T(X\theta-Y)+\alpha||\theta||_1$$

其中n 为样本数,$\alpha$为常数系数,需要进行调优,甚至还是一些绝对值较小的系数直接变为0,因此特别适用于参数数目缩减与参数的选择,因而用来估计稀疏参数的线性模型.

但是Lasso回归有一个很大的问题，导致我们需要把它单独拎出来讲，就是它的损失函数不是连续可导的，由于L1范数用的是绝对值之和，导致损失函数有不可导的点。也就是说，我们的最小二乘法，梯度下降法，牛顿法与拟牛顿法对它统统失效了。那我们怎么才能求有这个L1范数的损失函数极小值呢？