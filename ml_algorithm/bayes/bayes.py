#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-29 23:05:08
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import numpy as np

def loadDataSet():
	postingList = [
		['my', 'dog', 'has', 'flea', 'problems',
		'help', 'please'],
		['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
		['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him'],
		['stop', 'posting','stupid', 'worthless', 'garbage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
		['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
	]
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec
	pass

def createVocabList(dataSet):
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = vocabSet | set(doc)
	return list(vocabSet)
	pass

def setOfWord2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" %  word)
	return returnVec
	pass

def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1vect = np.log(p1Num/p1Denom)
	p0vect = np.log(p0Num/p0Denom)
	return p0vect, p1vect, pAbusive


def classifyNB(vec2classify, p0vec, p1vec, pclass1):
	p1 = sum(vec2classify * p1vec) + np.log(pclass1)
	p0 = sum(vec2classify * p0vec) + np.log(1.0 - pclass1)

	print('p1:', 1, p1)
	print('p0:', 0, p0)

	if p1 > p0:
		return 1
	else:
		return 0
	pass


def testNB():
	listpost, listclass = loadDataSet()
	myvocablist = createVocabList(listpost)
	trainMat = []
	for post in listpost:
		trainMat.append(setOfWord2Vec(myvocablist, post))

	p0v, p1v, pab = trainNB0(np.array(trainMat), np.array(listclass))
	test = ['love', 'my', 'dalmation']
	thisdoc = np.array(setOfWord2Vec(myvocablist, test))
	print(test, ' classified as : ', classifyNB(thisdoc, p0v, p1v, pab))
	test = ['stupid', 'garbage']
	thisdoc = np.array(setOfWord2Vec(myvocablist, test))
	print(test, 'classified as : ', classifyNB(thisdoc, p0v, p1v, pab))

	pass











