import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

import matplotlib.pyplot as plt
data_dir = '/Users/rick/src/ml_data/data/gbdt/'

train = pd.read_csv(data_dir + 'train_modified.csv')
target = 'Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target, IDcol]]
# print(x_columns)
X = train[x_columns]
y = train['Disbursed']


# # default
# rf0 = RandomForestClassifier(oob_score=True, random_state=10)
# rf0.fit(X, y)
# print(rf0.oob_score_)
# y_predprob = rf0.predict_proba(X)[:, 1]
# print('AUC Score(Train): %f' % metrics.roc_auc_score(y, y_predprob))

# param_test1 = {'n_estimators': list(range(10, 71, 10))}
# gsearch1 = GridSearchCV(
# 	estimator=RandomForestClassifier(
# 		min_samples_split=100,
# 		min_samples_leaf=20,
# 		max_depth=8,
# 		max_features='sqrt',
# 		random_state=10
# 	),param_grid=param_test1,
# 	scoring='roc_auc',
# 	cv=5
# )
# gsearch1.fit(X, y)
# print('grid_scores_')
# print(gsearch1.grid_scores_)
# print('best_params_')
# print(gsearch1.best_params_)
# print('best_score_')
# print(gsearch1.best_score_)

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 clause
#
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest
#
# iris = load_iris()
#
# X, y = iris.data, iris.target
#
# # This dataset is way too high-dimensional. Better do PCA:
# pca = PCA(n_components=2)
#
# # Maybe some original features where good, too?
# selection = SelectKBest(k=1)
#
# # Build estimator from PCA and Univariate selection:
#
# combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#
# # Use combined features to transform dataset:
# X_features = combined_features.fit(X, y).transform(X)
#
# svm = SVC(kernel="linear")
#
# # Do grid search over k, n_components and C:
#
# pipeline = Pipeline([("features", combined_features), ("svm", svm)])
#
# param_grid = dict(features__pca__n_components=[1, 2, 3],
#                   features__univ_select__k=[1, 2],
#                   svm__C=[0.1, 1, 10])
#
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
# grid_search.fit(X, y)
# print(grid_search.best_estimator_)