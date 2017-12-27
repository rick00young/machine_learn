import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

df_train = pd.read_csv('/Users/rick/src/ml_data/raw/search_relevance/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('/Users/rick/src/ml_data/raw/search_relevance/test.csv', encoding="ISO-8859-1")
df_desc = pd.read_csv('/Users/rick/src/ml_data/raw/search_relevance/product_descriptions.csv')

# print(df_train.head())
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
print(df_all.shape)

df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

stemmer = SnowballStemmer('english')


def str_stemmer(s):
	print('str_stemmer - ', s)
	return " ".join([stemmer.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
	print('str_common_word - ', str1)
	return sum(int(str2.find(word) >= 0) for word in str1.split())


df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)

df_all['commons_in_title'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_title']), axis=1)
df_all['commons_in_desc'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_description']), axis=1)
df_all = df_all.drop(['search_term', 'product_title', 'product_description'], axis=1)

df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]

test_ids = df_test['id']

y_train = df_train['relevance'].values

X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

params = [1, 3, 5, 6, 7, 8, 9, 10]
test_scores = []
for param in params:
	clf = RandomForestRegressor(n_estimators=30, max_depth=param)
	test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
	test_scores.append(np.mean(test_score))
	print('test_score:', test_score)

import matplotlib.pyplot as plt

# %matplotlib inline
plt.plot(params, test_scores)
plt.title("Param vs CV Error")
plt.show()
