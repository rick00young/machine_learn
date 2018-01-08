from sklearn import datasets
from gensim.parsing.preprocessing import STOPWORDS
import gensim
import re
import sys

news_dataset = datasets.fetch_20newsgroups(subset='all',
										   remove=("headers", "footers", "quotes"))
documents = news_dataset.data
print(documents[0])


def tokenize(text):
	text = text.lower()
	words = re.sub('\W', ' ', text).split()
	words = [w for w in words if w not in STOPWORDS]
	return words


processed_docs = [tokenize(doc) for doc in documents]
# print(processed_docs)
# sys.exit(1)
word_count_dict = gensim.corpora.Dictionary(processed_docs)
# print(word_count_dict)

word_count_dict.filter_extremes(no_below=20, no_above=0.1)
bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]

# print(bag_of_words_corpus)
# print(word_count_dict)
# lda

lda_model = gensim.models.LdaModel(corpus=bag_of_words_corpus,
								   num_topics=10,
								   id2word=word_count_dict,
								   passes=5)
# 使用并行LDA加快处理速度
# gensim.models.ldamulticore.LdaMulticore(corpus=None,
# 										num_topics=100,
# 										id2word=None,
# 										workers=None,
# 										chunksize=2000,
# 										passes=1,
# 										batch=False,
# 										alpha='symmetric',
# 										eta=None,
# 										decay=0.5,
# 										offset=1.0,
# 										eval_every=10,
# 										iterations=50,
# 										gamma_threshold=0.001,
# 										random_state=None)
# lda_model.print_topics(10)