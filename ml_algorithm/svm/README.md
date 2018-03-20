## SVM支持向量机
## SVM支持向量机
[http://www.cnblogs.com/pinard/p/6097604.html](http://www.cnblogs.com/pinard/p/6097604.html)
### 1. 回顾感知机模型
$$\left.{\sum_{x_i\in M}-y^{(i)}(w^Tx^{(i)}}+b)\middle/||w||_2\right.$$

### 2. 函数间隔与几何间隔
函数间隔:
$$r^′=y(w^Tx+b)$$

几何间隔:
$$r=\frac{y(w^Tx+b)}{||w||_2}=\frac{r^′}{||w||_2}$$


### 4. SVM模型目标函数与优化
SVM的模型是让所有点到超平面的距离大于一定的距离，也就是所有的分类点要在各自类别的支持向量两边。用数学式子表示为:

$$\begin{array}{ll}
\text{max}  & r=\frac{y(w^Tx+b)}{||w||_2}  \\
\text{s.t.} & y_i(w^Tx_i+b)= r^{′(i)} \ge r^{′(i)} (i=1,2,...m)\\
\end{array}$$

一般我们都取函数间隔$r^′$为1，这样我们的优化函数定义为：
$$\begin{array}{ll}
\text{max}  & r=\frac{y(w^Tx+b)}{||w||_2}  \\
\text{s.t.} & y_i(w^Tx_i+b)\ge 1(i=1,2,...m) \\
\end{array}$$

...引入拉格朗日乘子:
$$L(w,b,\alpha)=\frac{1}{2}||w||_2^2-\sum_{i=1}^{m}\alpha_i[y_i(w^Tx_i+b)-1] \space 满足\alpha_i\ge0$$

优化目标变成:

$$\underbrace{min}_{w,b}\underbrace{max}_{a_i\ge0}\text{ }L(w, b, \alpha)$$

对上式求偏导并定义优化函数:
$$\frac{\partial L}{\partial w}=0 => w=\sum_{i=1}^{m}\alpha_iy_ix_i$$
$$\frac{\partial L}{\partial b}=0 => \sum_{i=1}^{m}\alpha_iy_i=0$$


$$\psi(\alpha)=\underbrace{min}_{w,b}\space L(w, b, \alpha)$$

$$
\begin{align}
\psi(\alpha) & = \frac{1}{2}||w||_2^2-\sum_{i=1}^{m}\alpha_i[y_i(w^Tx_i+b)-1]  \tag{1}\\
 & =... \tag{2} \\
 & =\sum_{i=1}^{m}a_i-\frac{1}{2}\sum_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j \tag{3}
\end{align}
$$

等价极小化问题:
$$\underbrace{min}_{a}\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_j(x_i\cdot x_j) -\sum_{i=1}^{m}\alpha_i$$
$$s.t. \space \sum_{i=1}^{m}\alpha_iy_i=0$$
$$\alpha_i \ge 0 \space i=1,2,...m$$

##线性分类SVM面临的问题
线性可分SVM的学习方法对于非线性的数据集是没有办法使用的， 有时候不能线性可分的原因是线性数据集里面多了少量的异常点，由于这些异常点导致了数据集不能线性可分.

所谓的软间隔
SVM对训练集里面的每个样本$(x_i,y_i)$)引入了一个松弛变量$ξ_i≥0$,使函数间隔加上松弛变量大于等于1，也就是说：
$$y_i(w\cdot x_i+b)\ge1-\xi_i$$

软间隔最大化的SVM的学习条件:
$$min \frac{1}{2}||w||_2^2+C\sum_{i=1}^{m}\xi_i$$
$$s.t. \space y_i(w^Tx_i+b)\ge1-\xi_i \space (i=1,2,...m)$$
$$\xi_i\ge 0 \space (i=1,2...m)$$
C>0为处罚参数,可以理解为一般回归和分类问题正则化的参数,C越大,对误分类处罚越大.需要调参.

### 3.线性分类SVM的软间隔最大化目标函数的优
约束问题:
$$L(w,b,\xi,\alpha,\mu)=\frac{1}{2}||w||_2^2+C\sum_{i=1}^{m}\xi_i-\sum_{i=1}^{m}\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]-\sum_{i=1}^{n}\mu_i\xi_i$$
$$\mu_i\ge0, \alpha_i\ge0, 均为拉格朗日系数$$

原始优化目录:
$$\underbrace{min}_{w, b, \xi}\space \underbrace{max}_{\alpha_i\ge0, \mu_i\ge0} \space L(w, b, \alpha, \xi, \mu)$$

对偶问题为:
$$\underbrace{max}_{\alpha_i\ge0, \mu_i\ge0} \space \underbrace{min}_{w, b, \xi}\space L(w, b, \alpha, \xi, \mu)$$

求偏导:
$$\frac{\partial L}{\partial w}=0 => w=\sum_{i=1}^{m}\alpha_iy_ix_i$$
$$\frac{\partial L}{\partial b}=0 => \sum_{i=1}^{m}\alpha_iy_i=0$$
$$\frac{\partial L}{\partial \xi}=0 => C-\alpha_i-\mu_i=0$$

代入求得优化目标数学形式为:
$$\underbrace{max}_{\alpha}\sum_{i=1}^{m}\alpha_i-\frac{1}{2}\sum_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j$$
$$s.t. \space \sum_{i=1}^{m}\alpha_iy_i=0$$
$$C-\alpha_i-\mu_i=0 \tag{1}$$
$$\alpha_i\ge0\space(i=1,2,...m) \tag{2}$$
$$\mu_i\ge0\space(i=1,2,...m) \tag{3}$$

(1)(2)(2).,可以合并只剩下:
$$0\le\alpha_i\le C$$

## 线性不可分支持向量机与核函数
我们遇到线性不可分的样例时，常用做法是把样例特征映射到高维空间中去(如上一节的多项式回归）但是遇到线性不可分的样例，一律映射到高维空间，那么这个维度大小是会高到令人恐怖的。此时，核函数就体现出它的价值了，核函数的价值在于它虽然也是将特征进行从低维到高维的转换，但核函数好在它在低维上进行计算，而将实质上的分类效果（利用了内积）表现在了高维上，这样避免了直接在高维空间中的复杂计算，真正解决了SVM线性不可分的问题。

### 核函数
个函数要想成为正定核函数，必须满足他里面任何点的集合形成的Gram矩阵是半正定的。

#### 线性核函数
线性核函数（Linear Kernel）其实就是我们前两篇的线性可分SVM，表达式为：
$$K(x,z)=x\cdot z$$
　也就是说，线性可分SVM我们可以和线性不可分SVM归为一类，区别仅仅在于线性可分SVM用的是线性核函数。
#### 多项式核函数
多项式核函数（Polynomial Kernel）是线性不可分SVM常用的核函数之一，表达式为：
$$K(x,z)=(\gamma x\cdot z=r^d)$$
其中，$\gamma, r, d$都需要自己调参定义。

#### 高斯核 函数
　高斯核函数（Gaussian Kernel），在SVM中也称为径向基核函数（Radial Basis Function,RBF），它是非线性分类SVM最主流的核函数。libsvm默认的核函数就是它。表达式为：
$$K(x,z)=exp(-\gamma||x-z||^2)$$
其中，$\gamma$大于0，需要自己调参定义。

#### Sigmoid核函数
Sigmoid核函数（Sigmoid Kernel）也是线性不可分SVM常用的核函数之一，表达式为：
$$K(x,z)=tanh(\gamma x\cdot z+r)$$
其中，$\gamma, r$都需要自己调参定义。

#### 加核后约束
$$\underbrace{min}_{\alpha}\frac{1}{2}\sum_{j=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^{n}\alpha_i$$
$$s.t. \space \sum_{i=1}^{m}\alpha_i\alpha_j=0$$
$$0\le\alpha_i\le C$$

## SMO算法原理

## svm多分类
SVM解决多分类问题的方法 
SVM算法最初是为二值分类问题设计的，当处理多类问题时，就需要构造合适的多类分类器。目前，构造SVM多类分类器的方法主要有两类：一类是直接法，直接在目标函数上进行修改，将多个分类面的参数求解合并到一个最优化问题中，通过求解该最优化问题“一次性”实现多类分类。这种方法看似简单，但其计算复杂度比较高，实现起来比较困难，只适合用于小型问题中；另一类是间接法，主要是通过组合多个二分类器来实现多分类器的构造，常见的方法有one-against-one和one-against-all两种。 
 
 * a.一对多法（one-versus-rest,简称1-v-r SVMs）。训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。 
 * b.一对一法（one-versus-one,简称1-v-1 SVMs）。其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。Libsvm中的多类分类就是根据这个方法实现的。 
 * c.层次支持向量机（H-SVMs）。层次分类法首先将所有类别分成两个子类，再将子类进一步划分成两个次级子类，如此循环，直到得到一个单独的类别为止。 
对c和d两种方法的详细说明可以参考论文《支持向量机在多类分类问题中的推广》（计算机工程与应用。2004） 
* d.其他多类分类方法。除了以上几种方法外，还有有向无环图SVM（Directed Acyclic Graph SVMs，简称DAG-SVMs）和对类别进行二进制编码的纠错编码SVMs。