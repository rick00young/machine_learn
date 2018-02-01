import numpy as np


def loadDataSet(fileName):
	dataMat = []
	fr = open('/Users/rick/Documents/july_edu/machinelearninginaction/Ch09/' + fileName, 'r')
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		# print(curLine)
		fltLine = list(map(float, curLine))
		dataMat.append(fltLine)

	return dataMat


def binSplitDataSet(dataSet, feature, value):
	"""
	输入：数据集，数据集中某一特征列，该特征列中的某个取值
	功能：将数据集按特征列的某一取值换分为左右两个子数据集
	输出：左右子数据集
	"""
	mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
	mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
	return mat0, mat1


def regLeaf(dataSet):
	return np.mean(dataSet[:, -1])


def regErr(dataSet):
	# 由于回归树中用输出的均值作为叶节点，所以在这里求误差平方和实质上就是方差
	return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	# 当不满足阈值或某一子数据集下输出全相等时，返回叶节点
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
	# thresholdErr ops[0]
	# thresholdSamples ops[1]
	tolS = ops[0]
	tolN = ops[1]
	if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m, n = np.shape(dataSet)
	S = errType(dataSet)
	bestS = np.inf
	bestIndex = 0
	bestValue = 0
	for featIndex in range(n - 1):
		for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
				continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	# 检验在所选出的最优划分特征及其取值下，误差平方和与未划分时的差是否小于阈值，若是，则不适合划分
	# 这里可以看作是预剪枝 prepruning
	if (S - bestS) < tolS:
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	# 检验在所选出的最优划分特征及其取值下，划分的左右数据集的样本数是否小于阈值，若是，则不适合划分
	if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
		return None, leafType(dataSet)

	return bestIndex, bestValue


def isTree(obj):
	return type(obj).__name__ == 'dict'


# 取节点平均值,对树进行塌陷处理
def getMean(tree={}):
	if isTree(tree.get('right', '')):
		tree['right'] = getMean(tree['right'])
	if isTree(tree.get('left', '')):
		tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
	if np.shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
	if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)
	# if they are now both leafs, see if we can merge them
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
		               np.sum(np.power(rSet[:, -1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
		if errorMerge < errorNoMerge:
			print("merging")
			return treeMean
		else:
			return tree
	else:
		return tree


def linearSolve(dataSet):
	m, n = np.shape(dataSet)
	X = np.mat(np.ones((m,n)))
	Y = np.mat(np.ones((m, 1)))
	X[:, 1:n] = dataSet[:, 0:n-1]
	Y = dataSet[:, -1]
	xTx = X.T*X
	if np.linalg.det(xTx) == 0.0:
		raise NameError('This matrix is singular, cannot do reverse,'
		                'try increasing the second value of ops')
	ws = xTx.I * (X.T * Y)
	return ws, X, Y


def modelLeaf(dataSet):
	ws, X, Y = linearSolve(dataSet)
	# print(ws)
	return ws

def modelErr(dataSet):
	ws, X, Y = linearSolve(dataSet)
	yHat = X * ws
	return np.sum(np.power(Y - yHat, 2))

def regTreeEval(model, inData):
	return float(model)

def modelTreeEval(model, inData):
	n = np.shape(inData)[1]
	X = np.mat(np.ones((1, n+1)))
	X[:, 1:n+1] = inData
	return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
	if not isTree(tree):
		return modelEval(tree, inData)
	if inData[tree['spInd']] > tree['spVal']:
		if isTree(tree['left']):
			return treeForeCast(tree['left'], inData, modelEval)
		else:
			return modelEval(tree['left'], inData)
	else:
		if isTree(tree['right']):
			return treeForeCast(tree['right'], inData, modelEval)
		else:
			return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
	m = len(testData)
	yHat = np.mat(np.zeros((m, 1)))
	for i in range(m):
		yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
	return yHat

if '__main__' == __name__:
	# # mydata = loadDataSet(fileName='ex00.txt')
	# mydata2 = loadDataSet('ex2.txt')
	# myMat2 = np.mat(mydata2)
	# myTree = createTree(myMat2, ops=(0, 1))
	# print(myTree)
	# myDataTest = loadDataSet('ex2test.txt')
	# myMat2Test = np.mat(myDataTest)
	# prune(myTree, myMat2Test)
	# print(myTree)
	trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
	testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
	myTree = createTree(trainMat, ops=(1, 20))
	yHat = createForeCast(myTree, testMat[:, 0])
	print("corrcoef:", np.corrcoef(yHat, testMat[:, 1], rowvar=0))

	myTree = createTree(trainMat, modelLeaf, modelErr, (1,20))
	yHat =createForeCast(myTree, testMat[:, 0], modelTreeEval)
	print("corrcoef:", np.corrcoef(yHat, testMat[:, 1], rowvar=0))

	ws, x, y = linearSolve(trainMat)
	yHat = np.mat(np.zeros((np.shape(trainMat)[0], 1)))
	print(ws)
	for i in range(np.shape(testMat)[0]):
		yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]

	print("corrcoef:", np.corrcoef(yHat, testMat[:, 1], rowvar=0))
