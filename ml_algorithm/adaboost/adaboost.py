

import numpy as np
import matplotlib.pyplot as plt


def loadSimpleData():
	dataMat = np.matrix([
		[1., 2.1],
		[2., 1.1],
		[1.3, 1.0],
		[1., 1.],
		[2., 1.]
	])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return dataMat, classLabels


def pltData(dataMat, classLabels):
	for index, item in enumerate(dataMat):
		if classLabels[index] > 0:
			# or 表示画红点
			plt.plot(item[0, 0], item[0, 1], 'or')
		else:
			# ob 蓝点
			plt.plot(item[0, 0], item[0, 1], 'ob')
	plt.show()


# 通过比较阈值进行分类
# threshVal是阈值 threshIneq决定了不等号是大于还是小于
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 先全部设为1
	if threshIneq == 'lt':  # 然后根据阈值和不等号将满足要求的都设为-1
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	return retArray


# 在加权数据集里面寻找最低错误率的单层决策树
# D是指数据集权重 用于计算加权错误率
def buildStump(dataArr, classLabels, D):
	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T
	m, n = np.shape(dataMatrix)  # m为行数 n为列数
	numSteps = 10.0
	bestStump = {}
	bestClasEst = np.mat(np.zeros((m, 1)))
	minError = np.inf  # 最小误差率初值设为无穷大
	for i in range(n):  # 第一层循环 对数据集中的每一个特征 n为特征总数
		rangeMin = dataMatrix[:, i].min()
		rangeMax = dataMatrix[:, i].max()
		stepSize = (rangeMax - rangeMin) / numSteps
		for j in range(-1, int(numSteps) + 1):  # 第二层循环 对每个步长
			for inequal in ['lt', 'gt']:  # 第三层循环 对每个不等号
				threshVal = rangeMin + float(j) * stepSize  # 计算阈值
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 根据阈值和不等号进行预测
				errArr = np.mat(np.ones((m, 1)))  # 先假设所有的结果都是错的（标记为1）
				errArr[predictedVals == labelMat] = 0  # 然后把预测结果正确的标记为0
				weightedError = D.T * errArr  # 计算加权错误率
				# print('split: dim %d, thresh %.2f, thresh inequal: %s,he weightederror is %.3f' % (i,threshVal,inequal,weightedError))
				if weightedError < minError:  # 将加权错误率最小的结果保存下来
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst


# 基于单层决策树的AdaBoost训练函数
# numIt指迭代次数 默认为40 当训练错误率达到0就会提前结束训练
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
	weakClassArr = []  # 用于存储每次训练得到的弱分类器以及其输出结果的权重
	m = np.shape(dataArr)[0]
	D = np.mat(np.ones((m, 1)) / m)  # 数据集权重初始化为1/m
	aggClassEst = np.mat(np.zeros((m, 1)))  # 记录每个数据点的类别估计累计值
	for i in range(numIt):
		# 在加权数据集里面寻找最低错误率的单层决策树
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		print("D: ", D.T)
		# 根据错误率计算出本次单层决策树输出结果的权重 max(error,1e-16)则是为了确保error为0时不会出现除0溢出
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
		print("alpha:", alpha)
		bestStump['alpha'] = alpha  # 记录权重
		weakClassArr.append(bestStump)
		print('classEst: ', classEst.T)
		# 计算下一次迭代中的权重向量D
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 计算指数
		D = np.multiply(D, np.exp(expon))
		D = D / D.sum()  # 归一化
		# 错误率累加计算
		aggClassEst += alpha * classEst
		print('aggClassEst: ', aggClassEst.T)
		# print("D:", D.T)
		# aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
		# errorRate = aggErrors.sum()/m
		errorRate = 1.0 * sum(
			np.sign(aggClassEst) != np.mat(classLabels).T) / m  # sign(aggClassEst)表示根据aggClassEst的正负号分别标记为1 -1
		print('total error: ', errorRate)
		if errorRate == 0.0:  # 如果错误率为0那就提前结束for循环
			break
	return weakClassArr


# 基于AdaBoost的分类函数
# dataToClass是待分类样例 classifierArr是adaBoostTrainDS函数训练出来的弱分类器数组
def adaClassify(dataToClass, classifierArr):
	dataMatrix = np.mat(dataToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m, 1)))
	for i in range(len(classifierArr)):  # 遍历所有的弱分类器
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
		                         classifierArr[i]['thresh'], \
		                         classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		print(aggClassEst)
	return np.sign(aggClassEst)


# 自适应数据加载函数
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))
	dataMat = [];
	labelMat = []
	for line in open(fileName).readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat - 1):  # 最后一项为label
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat


# dataMat, classLabels = loadSimpleData()
# pltData(dataMat, classLabels)

if '__main__' == __name__:
	# dataMat, classLabels = loadSimpleData()
	# D = np.mat(np.ones((5, 1)) / 5)
	# buildStump(dataMat, classLabels, D)
	dataArr, labelArr = loadDataSet('/Users/rick/Documents/july_edu/machinelearninginaction/Ch07/horseColicTraining2.txt')
	classifierArray = adaBoostTrainDS(dataArr, labelArr, 10)

	# 测试
	# testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
	# prediction10 = adaClassify(testArr, classifierArray)
	# print(1.0 * sum(prediction10 != np.mat(testLabelArr).T) / len(prediction10))

