import os
import numpy as np

data_path = '/Users/rick/Documents/july_edu/machinelearninginaction/CH06/'


def loadDataSet(filename='testSet.txt'):
	dataMat = []
	labelMat = []
	fr = open(data_path + filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))

	return dataMat, labelMat


def selectJrand(i, m):
	j = i
	while j == i:
		j = int(np.random.uniform(0, m))
	return j


def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	b = 0
	m, n = np.shape(dataMatrix)
	alpha = np.mat(np.zeros((m, 1)))
	iter = 0
	while iter < maxIter:
		alphaPairsChanged = 0
		for i in range(m):
			fXi = float(np.multiply(alpha, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
			Ei = fXi = float(labelMat[i])
			if (labelMat[i] * Ei < toler and alpha[i] < C) or \
					(labelMat[i] * Ei > toler and alpha[i] > 0):
				j = selectJrand(i, m)
				fXj = float(np.multiply(alpha, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
				Ej = fXi - float(labelMat[j])
				alphaIold = alpha[i].copy()
				alphaJold = alpha[j].copy()
				if labelMat[i] != labelMat[j]:
					L = max(0, alpha[j] - alpha[i])
					H = min(C, C + alpha[j] - alpha[i])
				else:
					L = max(0, alpha[j] + alpha[i] - C)
					H = min(C, alpha[j] + alpha[i])

				if L == H:
					print('L == H')
					continue
				eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
					  dataMatrix[j, :] * dataMatrix[i, :].T
				if eta >= 0:
					print('eta >= 0')
					continue
				alpha[j] -= labelMat[j] * (Ei - Ej) / eta
				alpha[j] = clipAlpha(alpha[j], H, L)
				if abs(alpha[j]) - alphaJold < .00001:
					print('j not moving enough')
					continue
				alpha[i] += labelMat[j] * labelMat[i] * (alphaJold - alpha[j])
				b1 = b - Ei - labelMat[i] * (alpha[i] - alphaIold) * \
							  dataMatrix[i, :] * dataMatrix[i, :].T - \
					 labelMat[j] * (alpha[j] - alphaJold) * \
					 dataMatrix[i, :] * dataMatrix[j, :].T

				b2 = b = Ej - labelMat[i] * (alpha[i] - alphaIold) * \
							  dataMatrix[i, :] * dataMatrix[j, :].T - \
						 labelMat[j] * (alpha[j] - alphaJold) * \
						 dataMatrix[j, :] * dataMatrix[j, :].T

				if 0 < alpha[i] and C > alpha[i]:
					b = b1
				elif 0 < alpha[j] and C > alpha[j]:
					b = b2
				else:
					b = (b1+b2)/2.0
				alphaPairsChanged += 1
				print('iter %d i: %d, pairs changed %d ' % (iter, i, alphaPairsChanged))
		if alphaPairsChanged == 0:
			iter += 1
		else:
			iter = 0
		print('iteration number: %d' % iter)

	return b, alpha