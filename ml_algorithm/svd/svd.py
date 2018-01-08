import numpy as np
from numpy import linalg


def loadExData():
	return [[0, 0, 0, 2, 2],
			[0, 0, 0, 3, 3],
			[0, 0, 0, 1, 1],
			[1, 1, 1, 0, 0],
			[2, 2, 2, 0, 0],
			[5, 5, 5, 0, 0],
			[1, 1, 1, 0, 0]]


def loadExData2():
	return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
			[0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
			[0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
			[3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
			[5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
			[0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
			[4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
			[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
			[0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
			[0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
			[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA, inB):
	return 1.0/(1.0 * linalg.norm(inA - inB))


def pearsSim(inA, inB):
	if len(inA) < 3: return 1.0
	return .5 + .5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
	num = float(inA.T*inB)
	denom = linalg.norm(inA) * linalg.norm(inB)
	return .5+.5*(num/denom)


def standEst(dataMat, user, simMeans, item):
	n = np.shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0:continue
		overlap = np.nonzero(np.logical_and(dataMat[:, item].A > 0,
											dataMat[:, j].A > 0))[0]
		if len(overlap) == 0: similarity = 0
		else:
			similarity = simMeans(dataMat[overlap, item],
								  dataMat[overlap, j])
		print('the %d and %d similarity is : %f' % (item, j, similarity))
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal/simTotal


def recommend(dataMat, user, N=3, simMean=cosSim, estMethod=standEst):
	unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
	if len(unratedItems) == 0:
		return 'you rated everything.'

	itemScores = []
	for item in unratedItems:
		estimateScore = estMethod(dataMat, user, simMean, item)
		itemScores.append((item, estimateScore))
	return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]




def svdEst(dataMat, user, simMean, item):
	n = np.shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	u, sigma, vt = linalg.svd(dataMat)
	sig4 = np.mat(np.eye(4)*sigma[:4])
	# dataMat: (7, 5)
	# u: (7, 7) -> (7, 4)
	# sigma: (5,5)  -> (4,4)
	# vt: (5,5)
	xformedItems = dataMat.T * u[:, :4] * sig4.I
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0 or j == item:
			continue
		similarity = simMean(xformedItems[item, :].T,
							 xformedItems[j, :].T)
		print('the %d and %d similarity is: %f' % (item, j, similarity))
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal/simTotal


def printMat(inMat, thresh=.8):
	for i in range(32):
		for k in range(32):
			if float(inMat[i, k]) > thresh:
				print(1)
			else:
				print(0)
		print('')


def imgCompress(numSV=3, thresh=.8):
	my1 = []
	for line in open('0_5.txt').readlines():
		newRow = []
		for i in range(32):
			newRow.append(int(line[i]))
		my1.append(newRow)
	myMat = np.mat(my1)
	print('---original matrix---')
	printMat(myMat, thresh)
	u, sigma, vt = linalg.svd(myMat)
	sigRecon = np.mat(np.zeros((numSV, numSV)))
	for k in range(numSV):
		sigRecon[k, k] = sigma[k]
	reconMat = u[:, :numSV]*sigRecon*vt[:numSV, :]
	print('-----reconstruct matrix using %d singular values-----' % numSV)
	printMat(reconMat, thresh)