"""
### MCMC采样
	1. 蒙特卡罗方法
	2. 马尔科夫链
	3. MCMC采样和M-H采样
		M-H采样有两个缺点：
			一是需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长。
			二是有些高维数据，特征的条件概率分布好求，但是特征的联合分布不好求
	4. Gibbs采样
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.1)
y = x ** 2
plt.figure()
plt.plot(x, y)
plt.show()