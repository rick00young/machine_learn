## Ridge回归
由于直接套用线性回归可能产生过拟合，我们需要加入正则化项，如果加入的是L2正则化项，就是Ridge回归，有时也翻译为脊回归。它和一般线性回归的区别是在损失函数上增加了一个L2正则化的项，和一个调节线性回归项和正则化项权重的系数αα。损失函数表达式如下：
$$J(\theta)=\frac{1}{2}(X\theta-Y)^T(X\theta-Y) + \frac{1}{2}\alpha||\theta||_2^2$$

其中的$\alpha$为常数系数,需要进行调优.$||\theta||_2$为L2的范数

Ridge的回归的解法和一般线性回归大同小异,如果采用梯度下降法,则每一轮的$\theta$的迭代的表达式为:
$$\theta=\theta-(\beta X^T(X\theta-Y))+\alpha\theta$$
其中$\beta$为步长

如果用最小二痴乘法,则$\theta$的结果是:
$$\theta=(X^TX+\alpha E)^{-1}X^TY$$
其中E为单位距阵.

Ridge回归在不抛弃任何一个变量的情况下，缩小了回归系数，使得模型相对而言比较的稳定，但这会使得模型的变量特别多，模型解释性差