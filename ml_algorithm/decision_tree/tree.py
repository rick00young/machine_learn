import os
import operator
import math
import matplotlib.pyplot as plt
import matplotlib
import pickle

def calShannonEnt(dataSet):
	"""
	获取数据集的信息熵
	:param dataSet:
	:return:
	"""
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts:
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * math.log(prob, 2)
	return shannonEnt

def createDataSet():
	dataSet = [
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

def splitDataSet(dataSet, axis, value):
	"""
	按给定数据的特征位置及值对数据集进行划分
	:param dataSet:
	:param axis:
	:param value:
	:return:
	"""
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""
	寻找数据的最大信息增益参数
	:param dataSet:
	[
		[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']
	]
	:return:
	"""
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	# 数据特征遍历
	for i in range(numFeatures):
		# 获取数据集的某一个特征,组成新的数据集
		featList = [example[i] for example in dataSet]
		# 对特征值去重
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			# 用新数据的某一个特征对数据进行划分
			subDataSet = splitDataSet(dataSet, i, value)
			# 划分后子数据集的概率
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		print('infoGain:', infoGain)
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	print('bestInfoGain:', bestInfoGain, 'bestFeature:', bestFeature)
	return bestFeature


def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	# 如果类别完全相同,则停止划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 如果只剩下一个特征,数据还没有分类,则遍历所有特征,返回出现次数最多的
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	print(bestFeatLabel)
	myTree = {bestFeatLabel: {}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		print(value)
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

	return myTree

# decisionNode = dict(boxstyle='sawtooth', fc='0.8')
# leftNode = dict(boxstyle='round4', fc='0.8')
# arrow_args = dict(arrowstyle='<-')
#
# def plotNode(nodeTxt, centerPt, parentPt, nodeType):
# 	pass
def classify(inputTree, featLabels, testVec):
	firstStr  = [x for x in inputTree.keys()][0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel


def storeTree(inputTree, filename):
	fw = open(filename, 'wb')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	fr = open(filename, 'rb')
	return pickle.load(fr)



glass_data_path = '/Users/rick/Documents/july_edu/machinelearninginaction/Ch03/'

def loadGlassData():
	filename = 'lenses.txt'
	fr = open(glass_data_path + filename)
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = createTree(lenses, lensesLabels)
	print(lensesTree)
	return lensesTree

if '__main__' == __name__:
	loadGlassData()

