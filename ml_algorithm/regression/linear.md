## 线性回归
线性回归的一般形式:
$$h_\theta(X)=X\theta$$

需要极小化损失函数:
$$J(\theta)=\frac{1}{2}(X\theta-Y)^T(X\theta-Y)$$

如果用梯度下降法求解, 则每一轮$\theta$迭代的表达式为:
$$\theta=\theta-\alpha{X^T(X\theta-Y)}$$

如果用最小二乘法,则$\theta$的结果是:
$$\theta=(X^TX)^{-1}X^TY$$