import numpy as np
import operator
import matplotlib
import os
import matplotlib.pyplot as plt

def createDataSet():
	group = np.array([[1.0, 1.1], [1.0, 1], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inx, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distance = sqDistances**.5
	sortedDistance = distance.argsort()
	classCount = {}
	for i in range(k):
		voteLabel = labels[sortedDistance[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = np.zeros((numberOfLines, 3))
	classLabelVector =[]
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		# print(listFromLine[0:3])
		returnMat[index, :] = listFromLine[0:3]
		# print(returnMat[index, :])
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector


def autoNorm(dataSet):
	"""
	new_value = (old_value - min)/(max - min)
	:param dataSet:
	:return:
	"""
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(dataSet.shape)
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	return normDataSet, ranges, minVals


file_path = '/Users/rick/Documents/july_edu/machinelearninginaction/ch02/datingTestSet2.txt'

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix(file_path)
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]

	numTestVecs = int(m *  hoRatio)
	errorCount = 0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m], datingLabels[numTestVecs:m], 3)
		print('the classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))

		if classifierResult != datingLabels[i]:
			errorCount += 1
	print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small does', 'in large does']
	percentTats = float(input("percent of time spent playing video games?"))
	print('')
	ffMiles = float(input('frequent flier miles earned per year?'))
	print('')
	iceCream = float(input('liters of ice cream consumed per year?'))
	datingDataMat, datingLabels = file2matrix(file_path)
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = np.array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print('You will probably like this person:', resultList[classifierResult - 1])


img_train_path = '/Users/rick/Documents/july_edu/machinelearninginaction/Ch02/trainingDigits/'
img_test_path = '/Users/rick/Documents/july_edu/machinelearninginaction/Ch02/testDigits/'

def img2vector(filename):
	returnVec = np.zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVec[0, 32*i+j] = int(lineStr[j])
	return returnVec


def handwritingClassTest():
	hwLabels = []
	trainFileList = os.listdir(img_test_path)
	m = len(trainFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector(img_train_path + fileNameStr)

	testFileList = os.listdir(img_test_path)
	error_count = 0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector(img_test_path + fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print('the classifier came back with: %d, the real answer is: %d', (classifierResult, classNumStr))
		if classifierResult != classNumStr:
			error_count += 1
	print('the total number of error is :', error_count)
	print('the total error rate is: ', error_count/float(mTest))


if '__main__' == __name__:
	datingDataMat, datingLabels = file2matrix(file_path)
	# print(datingDataMat[0])
	# print(datingDataMat[1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
	plt.show()