'''
有如下图
	A->C, A->B, A->D,
	B->A, B->D
	C->
	D->B, D->C

  | A 	B 	C 	D
-----------------------
A |	0 	1/2 0 	0
B | 1/3 0 	0 	1/2
C |	1/3 0 	1 	1/2
D |	1/3	1/2	0 	0

 | 	1/4
 |	1/4
 |	1/4
 |	1/4

可以发现C是一个爬虫陷阱，因为一进入C就再也出不去了，因此如果用原始的PageRank算法，
收敛后到达C的概率为1，而到达A、B、D的概率均为0。
而增加随机跳转的机制之后，假设我们取β=0.8
vn=βMv[n−1]+(1−β)e/N
'''

import numpy as np
M = [
	[0, 	1/2, 	0, 	0],
	[1/3, 	0, 		0, 	1/2],
	[1/3, 	0, 		1, 	1/2],
	[1/3, 	1/2, 	0, 	0]
]

v0 = [[1/4],
	  [1/4],
	  [1/4],
	  [1/4]]

m_matrix = np.mat(M)
v_matrix = np.mat(v0)
beta = 0.8

for i in range(5):
	vn = beta*m_matrix.dot(v0) + (1-beta)*np.mat(v_matrix)

	print(vn)
print('--'*20)
print(vn)
# print(vn*148)