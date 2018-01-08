import numpy as np
def test1():
	"""
	从第60轮开始，我们的状态概率分布就不变了，一直保持在[0.625   0.3125  0.0625]
	:return:
	"""
	matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=float)
	vector1 = np.matrix([[0.3,0.4,0.3]], dtype=float)
	for i in range(100):
		vector1 = vector1*matrix
		print(vector1)


def test2():
	"""

	:return:
	"""
	matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
	vector1 = np.matrix([[0.7, 0.1, 0.2]], dtype=float)
	for i in range(100):
		vector1 = vector1 * matrix
		print("Current round:", i + 1)
		print(vector1)

# 可以看出，尽管这次我们采用了不同初始概率分布，最终状态的概率分布趋于同一个稳定的概率分布
#  也就是说我们的马尔科夫链模型的状态转移矩阵收敛到的稳定概率分布与我们的初始状态概率分布无关。这是一个非常好的性质


def test3():
	"""
	同时，对于一个确定的状态转移矩阵PP，它的n次幂PnPn在当n大于一定的值的时候也可以发现是确定的
	:return:
	"""
	matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
	for i in range(10):
		matrix = matrix * matrix
		print("Current round:", i + 1)
		print(matrix)


