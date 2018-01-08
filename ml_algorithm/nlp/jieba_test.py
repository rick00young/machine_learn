import jieba


jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)


with open('data/nlp_test2.txt') as f:
	document = f.read()
	document_cut = jieba.cut(document)
	# print(list(document_cut))
	result = ' '.join(list(document_cut))
	print(result)
	with open('data/nlp_test3.txt', 'w') as f2:
		f2.write(result)

f2.close()

f.close()
