'''
频繁项集(frequent item sets)是经常出现在一块的物品的集合,
关联规则(association rules)暗示两种物品之间可能存在很强的关系。

1-一个项集的支持度(support)
被定义为数据集中包含该项集的记录所占的比例。支持度是针对项集来说的,因此可以定义一个最小支持度,而只保留满足最
小支持度的项集。

2-可信度或置信度(confidence)
是针对一条诸如{尿布} ➞ {葡萄酒}的关联规则来定义的。这
条规则的可信度被定义为“支持度({尿布, 葡萄酒})/支持度({尿布})”。从图11-1中可以看到,由
于{尿布, 葡萄酒}的支持度为3/5,尿布的支持度为4/5,所以“尿布 ➞ 葡萄酒”的可信度为3/4=0.75。
这意味着对于包含“尿布”的所有记录,我们的规则对其中75%的记录都适用。


Apriori原理是说如果某个项集是频繁的,那么它的所有子集也是频繁的。这个原理直观上并没有什么帮助,
但是如果反过来看就有用了,也就是说如果一个项集是非频繁集,那么它的所有超集也是非频繁的。


当集合中项的个数大于0时:
    构建一个k个项组成的候选项集的列表
    检查数据以确认每个项集都是频繁的
    保留频繁项集并构建k+1项组成的候选项集的列表(向上合并)


一条规则P ➞ H的可信度定义为 support(P |H)/support(P)

假设规则0,1,2 ➞ 3并不满足最小可信度要求,那么就知道任何左部为{0,1,2}子集的规则也不会满足最小可信度要求。
可以利用关联规则的上述性质属性来减少需要测试的规则数目。
'''

import numpy as np

def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
	C1 =[]
	for transaction in dataSet:
		for item in transaction:
			if item not in C1:
				C1.append(item)

	C1.sort()
	print(C1)
	return list(map(lambda x: frozenset([x]), C1))

#该函数用于从 C1 生成 L1 。
def scanD(D, CK, minSupport):
	# 参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
	ssCnt = {}
	for tid in D:
		for can in CK:
			# 判断候选项中是否含数据集的各项
			if can.issubset(tid):
				if can not in ssCnt:
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

# total apriori
# 组合，向上合并
def aprioriGen(Lk, k):
	# creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
	retList = []
	lenLk = len(Lk)

	for i in range(lenLk):
		# 两两组合遍历
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()
			# 若两个集合的前k-2个项相同时,则将两个集合合并
			if L1 == L2:
				# set union
				retList.append(Lk[i] | Lk[j])
	return retList


# aprioir
def apriori(dataSet, minSupport=0.5):
	C1 = createC1(dataSet)
	D = list(map(set, dataSet))
	# 单项最小支持度判断 0.5，生成L1
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	# 创建包含更大项集的更大列表,直到下一个大的项集为空
	while(len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData

#生成关联规则
def generateRules(L, supportData, minConf=0.7):
	# 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
	# 存储所有的关联规则
	bigRuleList = []
	# 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
	for i in range(1, len(L)):
		# 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
		for freqSet in L[i]:
			# 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
			H1 = [frozenset([item]) for item in freqSet]
			if i > 1:
				# 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				# 第一层时，后件数为1
				calConf(freqSet, H1, supportData, bigRuleList, minConf)
	return  bigRuleList

#生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calConf(freqSet, H, supportData, br1, minConf=0.7):
	# 针对项集中只有两个元素时，计算可信度
	# 返回一个满足最小可信度要求的规则列表
	# 后件，遍历 H中的所有项集并计算它们的可信度值
	pruneH = []
	for conseq in H:
		# 可信度计算，结合支持度数据
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >=minConf:
			print(freqSet, '::',conseq, '===', freqSet-conseq, conf)
			print(freqSet - conseq, '-->', conseq, 'conf:', conf)
			# 如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
			# 添加到规则里，brl 是前面通过检查的 bigRuleList
			br1.append((freqSet-conseq, conseq, conf))
			# 同样需要放入列表到后面检查
			pruneH.append(conseq)
	return pruneH


#合并
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
	# 参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
	m = len(H[0])
	# 频繁项集元素数目大于单个集合的元素数
	if len(freqSet) > (m + 1):
		Hmp1 = aprioriGen(H, m+1)
		Hmp1 = calConf(freqSet, Hmp1, supportData, br1, minConf)
		if (len(Hmp1) > 1):
			# 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
			rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

