'''
将电影评论转换成csv
电影评论地址: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
'''

import pyprind
import pandas as pd
import os
import numpy as np
import re

movie_csv = '/Users/rick/src/ml_data/data/aclImdb_data/movie_data.csv'
movie_pure_csv = '/Users/rick/src/ml_data/data/aclImdb_data/movie_pure_data.csv'

if not os.path.exists('/Users/rick/src/ml_data/data/aclImdb_data/movie_data.csv'):
	np.random.seed(0)
	pbar = pyprind.ProgBar(50000)

	labels = {'pos': 1, 'neg': 0}
	df = pd.DataFrame()
	for s in ['test', 'train']:
		for l in ['pos', 'neg']:
			path = '/Users/rick/src/ml_data/raw/aclImdb_data/aclImdb/%s/%s' % (s, l)
			for file in os.listdir(path):
				with open(os.path.join(path, file), 'ocr') as infile:
					txt = infile.read()
				df = df.append([[txt, labels[l]]], ignore_index=True)
				pbar.update()


	df.columns = ['review', 'sentiment']

	df = df.reindex(np.random.permutation(df.index))

	df.to_csv(movie_csv)
	df.head(3)
	print('done!')
else:
	print('file still exists!')

