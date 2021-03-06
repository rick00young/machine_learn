import numpy as np


def loadDataSet(fileName):
	fr = open('/Users/rick/Documents/july_edu/machinelearninginaction/Ch10/' + fileName, 'ocr')
	dataMat = []
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float, curLine))
		dataMat.append(fltLine)
	return dataMat


def distEclud(vecA, vecB):
	return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
	n = np.shape(dataSet)[1]
	centroids = np.mat(np.zeros((k, n)))
	for j in range(n):
		minJ = np.min(dataSet[:, j])
		rangeJ = float(np.max(dataSet[:, j]) - minJ)
		centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))

	return centroids


def kMeans(dataSet, k, distMean=distEclud, createCent=randCent):
	m = np.shape(dataSet)[0]
	clusterAssment = np.mat(np.zeros((m, 2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True

	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = np.inf
			minIndex = -1
			for j in range(k):
				distJI = distMean(centroids[j, :], dataSet[i, :])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist ** 2
		print(centroids)
		for cent in range(k):
			ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
			centroids[cent, :] = np.mean(ptsInClust, axis=0)

		return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
	m = np.shape(dataSet)[0]
	clusterAssment = np.mat(np.zeros((m, 2)))
	centroid0 = np.mean(dataSet, axis=0).tolist()[0]
	centList = [centroid0]  # create a list with one centroid
	for j in range(m):  # calc initial Error
		clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
	while (len(centList) < k):
		lowestSSE = np.inf
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0],
			                   :]  # get the data points currently in cluster i
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
			sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
			sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
			print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit
		bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
		bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		print('the bestCentToSplit is: ', bestCentToSplit)
		print('the len of bestClustAss is: ', len(bestClustAss))
		centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
		centList.append(bestNewCents[1, :].tolist()[0])
		clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
		:] = bestClustAss  # reassign new clusters, and SSE
	return np.mat(centList), clusterAssment


if '__main__' == __name__:
	dataMat = np.mat(loadDataSet('testSet.txt'))
	# print(dataMat)
