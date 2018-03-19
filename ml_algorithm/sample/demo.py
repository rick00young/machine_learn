"""
### MCMC采样
	1. 蒙特卡罗方法
	2. 马尔科夫链
		马尔科夫链的收敛性质
			1. 非周期
			2. 任何两个状态是连通的
			3. 马尔科夫链状态数是有限或无限的
			4. pi通常称为马尔科夫链的平稳分布
	3. MCMC采样和M-H采样
		http://www.cnblogs.com/pinard/p/6638955.html
		M-H采样有两个缺点：
			一是需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长。
			二是有些高维数据，特征的条件概率分布好求，但是特征的联合分布不好求
	4. Gibbs采样
"""


# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
#
# x = np.arange(0, 1, 0.1)
# y = x ** 2
# plt.figure()
# plt.plot(x, y)
# plt.show()


# M-H采样
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
# %matplotlib inline

def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
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