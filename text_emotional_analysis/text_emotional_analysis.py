import pandas as pd
import os
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()
stop = stopwords.words('english')


def preprocessor(text):
	text = re.sub('<[^>]*>', '', text)
	emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = re.sub('[\W]+', ' ', text.lower()) + ','.join(emotions).replace('-', '')
	# print(emotions)
	return text


def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]


def tokenizer(text):
	return text.split()


movie_csv = '/Users/rick/src/ml_data/data/aclImdb_data/movie_data.csv'
df = pd.read_csv(movie_csv)
# test = preprocessor('</a>this :) is :( a test :-) !<>')
# print(test)
# test = tokenizer_porter(text='runners like running and thus they run')
# print(test)
# test = [w for w in tokenizer_porter('runners like running and thus they run')[-10:] if w not in stop]
# print(test)

df['review'] = df['review'].apply(preprocessor)

x_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values

x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [
	{'vect__ngram_range': [(1, 1)],
	 'vect__stop_words': [stop, None],
	 'vect__tokenizer': [tokenizer_porter],
	 'clf__penalty': ['l1', 'l2'],
	 'clf__C': [1.0, 10.0, 100.0]},

	{'vect__ngram_range': [(1, 1)],
	 'vect__stop_words': [stop, None],
	 'vect__tokenizer': [tokenizer_porter],
	 'vect__use_idf': [False],
	 'vect__norm': [None],
	 'clf__penalty': ['l1', 'l2'],
	 'clf__C': [1.0, 10.0, 100.0]}
]
lr_tfidf = Pipeline([('vect', tfidf),
					 ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
						   scoring='accuracy',
						   cv=5, verbose=1,
						   n_jobs=-1)
gs_lr_tfidf.fit(x_train, y_train)

print('best param set %s' % gs_lr_tfidf.best_params_)

print('CV Accuracy: %3.f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % (clf.score(x_test, y_test)))