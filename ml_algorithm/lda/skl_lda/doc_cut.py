import jieba
import re
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
#第一个文档分词#

def preprocessor(text):
	text = re.sub('[，。；、]', '', text)
	return text

with open('data/nlp_test0.txt') as f:
	document = f.read()
	document_cut = list(jieba.cut(document))
	result = ' '.join(document_cut)
	result = preprocessor(result)
	print(result)
	with open('data/nlp_test1.txt', 'w') as f2:
		f2.write(result)

f.close()
f2.close()


with open('data/nlp_test2.txt') as f:
	document = f.read()
	document_cut = list(jieba.cut(document))
	result = ' '.join(document_cut)
	result = preprocessor(result)
	print(result)
	with open('data/nlp_test3.txt', 'w') as f2:
		f2.write(result)

f.close()
f2.close()

jieba.suggest_freq('桓温', True)
with open('data/nlp_test4.txt') as f:
	document = f.read()
	document_cut = list(jieba.cut(document))
	result = ' '.join(document_cut)
	result = preprocessor(result)
	print(result)
	with open('data/nlp_test5.txt', 'w') as f2:
		f2.write(result)

f.close()
f2.close()