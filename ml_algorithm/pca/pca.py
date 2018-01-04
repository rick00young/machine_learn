import numpy as np
from numpy import linalg


file_path = '/Users/rick/Documents/july_edu/machinelearninginaction/Ch13/'
def loadDataSet(filename='testSet.txt', delim='\t'):
	fr = open(file_path + filename)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	# print(stringArr)
	dataArr = [list(map(float, line)) for line in stringArr]
	return np.mat(dataArr)

def pca(dataMat, topNfeat=9999999):
	meanVals = np.mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals #remove mean
	covMat = np.cov(meanRemoved, rowvar=0)
	eigVals,eigVects = linalg.eig(np.mat(covMat))
	eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
	eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
	redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
	lowDDataMat = meanRemoved * redEigVects		#transform data into new dimensions
	print('meanRemoved:', meanRemoved.shape)
	print('redEigVects:', redEigVects.shape)
	print('lowDDataMat:', lowDDataMat.shape)
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat


# def replaceNanWithMean(filename='secom.data'):
# 	dataMat = loadDataSet(filename=filename)
# 	numFeat = np.shape(dataMat)[1]
# 	for i in range(numFeat):
# 		meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
# 		dataMat[np.nonzero(np.isnan(dataMat[:, i]))[0], i] = meanVal
#
	return dataMat

def replaceNanWithMean():
	datMat = loadDataSet('secom.data', ' ')
	numFeat = np.shape(datMat)[1]
	for i in range(numFeat):
		meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
		datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
	return datMat