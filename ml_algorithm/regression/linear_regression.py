import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
	fr = open('/Users/rick/Documents/july_edu/machinelearninginaction/Ch08/' + fileName, 'r')
	dataMat = []
	labelMat = []
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		lineArr = list(map(float, curLine))
		dataMat.append(lineArr[:-1])
		labelMat.append(lineArr[-1])
	return dataMat, labelMat


def standRegres(xArr, yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T * xMat
	if np.linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = xTx.I * (xMat.T * yMat)
	return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weight = np.mat(np.eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weight[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
	xTx = xMat.T * (weight * xMat)

	if np.linalg.det(xTx) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return
	ws = xTx.I * (xMat.T * (weight * yMat))
	return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i], xArr, yArr, k)
	return yHat


def rssError(yArr, yHatArr):
	return ((yArr - yHatArr) ** 2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print
		"This matrix is singular, cannot do inverse"
		return
	ws = denom.I * (xMat.T * yMat)
	return ws


def ridgeTest(xArr, yArr):
	xMat = np.mat(xArr);
	yMat = np.mat(yArr).T
	yMean = np.mean(yMat, 0)
	yMat = yMat - yMean  # to eliminate X0 take mean off of Y
	# regularize X's
	xMeans = np.mean(xMat, 0)  # calc mean then subtract it off
	xVar = np.var(xMat, 0)  # calc variance of Xi then divide by it
	xMat = (xMat - xMeans) / xVar
	numTestPts = 30
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
		wMat[i, :] = ws.T
	return wMat


def regularize(xMat):
	inMat = xMat.copy()
	inMeans = np.mean(inMat, 0)
	inVar = np.var(inMat, 0)
	inMat = (inMat - inMeans) / inVar
	return inMat


# 前向逐步线性加归
def stageWise(xArr, yArr, eps=0.01, numIt=100):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	yMean = np.mean(yMat, 0)
	yMat = yMat - yMean  # can also regularize ys but will get smaller coef
	xMat = regularize(xMat)
	m, n = np.shape(xMat)
	returnMat = np.zeros((numIt,n)) #testing code remove
	ws = np.zeros((n, 1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):
		print(ws.T)
		lowestError = np.inf
		for j in range(n):
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign
				yTest = xMat * wsTest
				rssE = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:]=ws.T
	return returnMat


if '__main__' == __name__:
	xArr, yArr = loadDataSet('ex0.txt')
	# print(xArr[0:2])
	# ws = standRegres(xArr, yArr)
	# print("ws:", ws)
	#
	# xMat = np.mat(xArr)
	# yMat = np.mat(yArr)
	# yHat = xMat*ws
	#
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
	# xCopy = xMat.copy()
	# xCopy.sort(0)
	# yHat = xCopy*ws
	# ax.plot(xCopy[:,1], yHat)
	# print(xCopy)
	# print("corrcoef:", np.corrcoef(yHat.T, yMat))
	# plt.show()


	# print(yArr[0])
	# test = lwlr(xArr[0], xArr, yArr, 1.0)
	# print('test:', test)
	# test = lwlr(xArr[0], xArr, yArr, 0.001)
	# print('test:', test)
	# yHat = lwlrTest(xArr, xArr, yArr, 0.003)
	# xMat = np.mat(xArr)
	# strInd = xMat[:, 1].argsort(0)
	# xSort = xMat[strInd][:, 0, :]
	# # print(xSort.shape)
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(xSort[:, 1], yHat[strInd])
	# ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
	# plt.show()

	# 预测鲍鱼年龄
	# abX, abY = loadDataSet('abalone.txt')
	# yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
	# yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
	# yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
	# print('error-yHat01:', rssError(abY[100:199], yHat01.T))
	# print('error-yHat1:', rssError(abY[100:199], yHat1.T))
	# print('error-yHat10:', rssError(abY[100:199], yHat10.T))
	#
	# ws = standRegres(abX[0:99], abY[0:99])
	# yHat = np.mat(abX[100:199]) * ws
	# print('stand error:', rssError(abY[100:199], yHat.T.A))

	# ridge
	# abX, abY = loadDataSet('abalone.txt')
	# ridgeWeight = ridgeTest(abX, abY)
	# print(ridgeWeight)
	# # print(np.mat(abX).shape)
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(ridgeWeight[:])
	# plt.show()


	abX, abY = loadDataSet('abalone.txt')
	stageWise(abX, abY, 0.001, 500)