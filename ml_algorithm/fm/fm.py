import math
import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import datetime


def loadDataSet():
	data = datasets.load_digits()
	train_x, test_x, train_y, test_y = train_test_split(data.data,
	                                                    data.target,
	                                                    test_size=0.3,
	                                                    random_state=1)
	return train_x, test_x, train_y, test_y


def sigmoid(inx):
	# return 1.0 / (1 + math.exp(-inx))
	return 1. / (1. + math.exp(-max(min(inx, 15.), -15.)))

def stocGradAscent(dataMatrix, classLabels, k, iter):
	# dataMatrix用的是mat, classLabels是列表
	m, n = np.shape(dataMatrix)
	alpha = 0.01
	# 初始化参数
	w = np.zeros((n, 1))  # 其中n是特征的个数
	w_0 = 0.
	v = random.normalvariate(0, 0.2) * np.ones((n, k))

	for it in range(iter):
		print(it)
		for x in range(m):  # 随机优化，对每一个样本而言的
			inter_1 = dataMatrix[x] * v
			inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  # multiply对应元素相乘
			# 完成交叉项
			interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.

			p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

			loss = sigmoid(classLabels[x] * p[0, 0]) - 1
			print(loss)

			w_0 = w_0 - alpha * loss * classLabels[x]

			for i in range(n):
				if dataMatrix[x, i] != 0:
					w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
					for j in range(k):
						v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
						dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

	return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
	m, n = np.shape(dataMatrix)
	allItem = 0
	error = 0
	result = []
	for x in range(m):
		allItem += 1
		inter_1 = dataMatrix[x] * v
		inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  # multiply对应元素相乘
		# 完成交叉项
		interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
		p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

		pre = sigmoid(p[0, 0])

		result.append(pre)

		if pre < 0.5 and classLabels[x] == 1.0:
			error += 1
		elif pre >= 0.5 and classLabels[x] == -1.0:
			error += 1
		else:
			continue

	print(result)

	return float(error) / allItem


if '__main__' == __name__:
	train_x, test_x, train_y, test_y = loadDataSet()
	# print(type(train_x))
	# print(type(train_y))
	date_startTrain = datetime.datetime.now()
	dataTrain = train_x
	labelTrain = train_y
	dataTest = test_x
	labelTest = train_y

	print("开始训练")
	w_0, w, v = stocGradAscent(np.mat(dataTrain), labelTrain, 20, 200)
	print("训练准确性为：%f" % (1 - getAccuracy(np.mat(dataTrain), labelTrain, w_0, w, v)))
	date_endTrain = datetime.datetime.now()
	print("训练时间为：%s" % (date_endTrain - date_startTrain))
	print("开始测试")
	print("测试准确性为：%f" % (1 - getAccuracy(np.mat(dataTest), labelTest, w_0, w, v)))





