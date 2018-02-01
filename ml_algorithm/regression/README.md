## 线性回归

### 局部加权线性回归:Locally Weighted Linear Regression , LWLR
使用核函数

高斯核:

$$\hat{w} = (X^TWX)^{-1}X^TWy$$

高斯核对的权重如下:

$$w(i,j)=exp\Biggl(\frac{|x^{(i)}-x|}{-2k^2}\Biggr)$$


缺点:

局部加权线性增加了计算量,它对每个点做预测时都必须使用整个数据集

