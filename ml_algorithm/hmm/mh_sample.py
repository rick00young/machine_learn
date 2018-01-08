"""
我们的目标平稳分布是一个均值3，标准差2的正态分布，而选择的马尔可夫链状态转移矩阵Q(i,j)Q(i,j)的条件转移概率是以ii为均值,方差1的正态分布在位置jj的值。这个例子仅仅用来让大家加深对M-H采样过程的理解。
毕竟一个普通的一维正态分布用不着去用M-H采样来获得样本。
:return:
"""
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt


"""
5. M-H采样总结
　　　　M-H采样完整解决了使用蒙特卡罗方法需要的任意概率分布样本集的问题，因此在实际生产环境得到了广泛的应用。

　　　　但是在大数据时代，M-H采样面临着两大难题：

　　　　1） 我们的数据特征非常的多，M-H采样由于接受率计算式π(j)Q(j,i)π(i)Q(i,j)π(j)Q(j,i)π(i)Q(i,j)的存在，在高维时需要的计算时间非常的可观，算法效率很低。同时α(i,j)α(i,j)一般小于1，有时候辛苦计算出来却被拒绝了。能不能做到不拒绝转移呢？

　　　　2） 由于特征维度大，很多时候我们甚至很难求出目标的各特征维度联合分布，但是可以方便求出各个特征之间的条件概率分布。这时候我们能不能只有各维度之间条件概率分布的情况下方便的采样呢？

　　　　Gibbs采样解决了上面两个问题，因此在大数据时代，MCMC采样基本是Gibbs采样的天下，下一篇我们就来讨论Gibbs采样。
"""

def norm_dist_prob(theta):
	y = norm.pdf(theta, loc=3, scale=2)
	return y

T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T - 1:
	t = t + 1
	pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
	alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))

	u = random.uniform(0, 1)
	if u < alpha:
		pi[t] = pi_star[0]
	else:
		pi[t] = pi[t - 1]

plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 50
plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
plt.show()