from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



stop_word_file = '/Users/rick/work_space/www/machine_learn/ml_algorithm/nlp/data/stop_words.txt'
stop_word_list = []
with open(stop_word_file, 'ocr') as f:
    stop_word_content = f.read()
    stop_word_list = stop_word_content.splitlines()
f.close()

# print(stop_word_list)



with open('data/nlp_test1.txt') as f3:
    res1 = f3.read()
print(res1)
print('-'*20)
with open('data/nlp_test3.txt') as f4:
    res2 = f4.read()
print(res2)
print('-'*20)
with open('data/nlp_test5.txt') as f5:
    res3 = f5.read()
print(res3)


corpus = [res1, res2, res3]
# print(corpus)
cntVector = CountVectorizer(stop_words=stop_word_list)
cntTf = cntVector.fit_transform(corpus)
# print(cntTf)

lda = LatentDirichletAllocation(n_topics=2,
								learning_offset=50.,
								random_state=0)
docRes = lda.fit_transform(cntTf)

print(docRes)