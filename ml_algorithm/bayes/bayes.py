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
	# p(x)的先验概率
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	# 初始化计数器
	p0Num = np.ones(numWords)
	p1Num = np.ones(numWords)
	# 防止概率太小发生下溢
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# 条件概率
	# log 是为防止
	p1vect = np.log(p1Num/p1Denom)
	p0vect = np.log(p0Num/p0Denom)
	return p0vect, p1vect, pAbusive


def classifyNB(vec2classify, p0vec, p1vec, pclass1):
	# p(a|b) = a(b|a)p(a)/p(b)
	# 进行概率大小比较时由于分母都一样,所以可以不计算分母,只计算分子
	# 此处相加是上式两取对数: ln(a*b) = ln(a) + ln(b)
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

def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	# print(returnVec)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec
	pass


def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []
	classList =[]
	fullText = []
	email_path = '/Users/rick/Documents/july_edu/machinelearninginaction/Ch04/'
	for i in range(1, 26):
		try:
			email = open(email_path + 'email/spam/%d.txt' % i).read()
			# print(email)
			wordList = textParse(email)
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(1)
		except Exception as e:
			print(e)

		print(i)
		try:
			email = open(email_path + 'email/ham/%d.txt' % i).read()
			wordList = textParse(email)
			docList.append(wordList)
			fullText.extend(wordList)
			classList.append(0)
		except Exception as e:
			print(e)

	vocabList = createVocabList(docList)
	trainingSet = [i for i in range(len(docList))]
	print('----', len(docList))
	# return
	testSet = []
	for i in range(10):
		randIndex = int(np.random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []
	trainClass = []
	for docIndex in trainingSet:
		trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
		trainClass.append(classList[docIndex])

	p0v, p1v, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
	errorCount = 0
	# error_doc = []
	for docIndex in testSet:
		wordVector = setOfWord2Vec(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
			errorCount += 1
			# error_doc.append(docList)

	print('the error rate is: ', float(errorCount)/len(testSet))


# spamTest()
def calMostFreq(vocabList, fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
	return sortedFreq[:30]

def localWords(feed1, feed0):
	import feedparser
	docList = []
	classList = []
	fullText = []
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30words = calMostFreq(vocabList, fullText)
	print(top30words)
	for pairW in top30words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	trainingSet = [x for x in range(2*minLen)]
	testSet = []
	for i in range(20):
		randIndex = int(np.random.uniform(len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []
	trainClass = []

	for docIndex in trainingSet:
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClass.append(classList[randIndex])

	p0v, p1v, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is: ', float(errorCount)/len(testSet))
	return vocabList, p0v, p1v


def getTopWords(ny, sf):
	import operator
	vocabList, p0v, p1v = localWords(ny, sf)
	topNY = []
	topSF = []
	for i in range(len(p0v)):
		if p0v[i] > -6.0: topSF.append((vocabList[i], p0v[i]))
		if p1v[i] > -6.0: topNY.append((vocabList[i], p1v[i]))

	sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
	print("SF**" * 10)
	for item in sortedSF:
		print(item[0])

	sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
	print('NY**' * 10)
	for item in sortedNY:
		print(item[0])



#test
if '__main__' == __name__:
	import feedparser
	ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
	sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
	vocabList, psf, pny = localWords(ny, sf)
	print(vocabList, psf, pny)

