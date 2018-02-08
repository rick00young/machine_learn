import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

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
# print(X)

# 不管任何参数，都用默认的
# gbm0 = GradientBoostingClassifier(random_state=10)
# gbm0.fit(X, y)
# y_pred = gbm0.predict(X)
# y_predprob = gbm0.predict_proba(X)
# print(y_predprob)
# print("Accuracy: %.4f" % metrics.accuracy_score(y.values, y_pred))
# print('AUC Score(train): %f' % metrics.roc_auc_score(y,y_predprob[:, 1]))

#从步长(learning rate)和迭代次数(n_estimators)入手,我们将步长初始值设置为0.1
# param_test1 = {'n_estimators': list(range(20, 81, 10))}
# gsearch1= GridSearchCV(estimator=GradientBoostingClassifier(
# 	learning_rate=0.1,
# 	min_samples_split=300,
# 	min_samples_leaf=20,
# 	max_features='sqrt',
# 	subsample=0.8,
# 	random_state=10
# ), param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
# gsearch1.fit(X, y)
# print('grid_scores_')
# print(gsearch1.grid_scores_)
# print('best_params_')
# print(gsearch1.best_params_)
# print('best_score_')
# print(gsearch1.best_score_)

#
param_test2 = {'max_depth': list(range(3,14,2)),
               'min_samples_split': list(range(100, 801, 200))}
gsearch1 = GridSearchCV(
	estimator=GradientBoostingClassifier(
		learning_rate=0.1,
		n_estimators=60,
		min_samples_leaf=20,
		max_features='sqrt',
		subsample=0.8,
		random_state=10
	),
	param_grid=param_test2,
	scoring='roc_auc',
	iid=False,
	cv=5
)
gsearch1.fit(X, y)
print('grid_scores_')
print(gsearch1.grid_scores_)
print('best_params_')
print(gsearch1.best_params_)
print('best_score_')
print(gsearch1.best_score_)