import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

stop_word_file = 'data/stop_words.txt'
stop_word_list = []
with open(stop_word_file, 'r') as f:
	stop_word_content = f.read()
	stop_word_list = stop_word_content.splitlines()
f.close()


with open('data/nlp_test1.txt') as f3:
	res1 = f3.read()
print(res1)
with open('data/nlp_test3.txt') as f4:
	res2 = f4.read()
print(res2)


corpus = [res1, res2]
vector = TfidfVectorizer(stop_words=stop_word_list)
tfidf = vector.fit_transform(corpus)

wordlist = vector.get_feature_names()#获取词袋模型中的所有词
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):
    print("-------第",i,"段文本的词语tf-idf权重------" )
    for j in range(len(wordlist)):
        print(wordlist[j],weightlist[i][j])


"""
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus=["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]

vectorizer=CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print(tfidf)



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
re = tfidf2.fit_transform(corpus)
print(re)
"""