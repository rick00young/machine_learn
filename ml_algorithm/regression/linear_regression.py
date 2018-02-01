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
	xTx = xMat.T*xMat
	if np.linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return
	ws = xTx.I*(xMat.T*yMat)
	return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weight = np.mat(np.eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weight[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
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
	return  yHat

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


	print(yArr[0])
	test = lwlr(xArr[0], xArr, yArr, 1.0)
	print('test:', test)
	test = lwlr(xArr[0], xArr, yArr, 0.001)
	print('test:', test)
	yHat = lwlrTest(xArr, xArr, yArr, 0.003)
	xMat = np.mat(xArr)
	strInd = xMat[:, 1].argsort(0)
	xSort = xMat[strInd][:, 0, :]
	# print(xSort.shape)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:, 1], yHat[strInd])
	ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
	plt.show()