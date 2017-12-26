import tflearn
import sys, os
import numpy as np
from sklearn.cross_validation import train_test_split
from termcolor import cprint

import load_face_feature
x, y, l = load_face_feature.load_feature()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
# print(x)
# print(y)
# sys.exit(1)


# Build neural network
net = tflearn.input_data(shape=[None, 128])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, len(l), activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(x_train, y_train, n_epoch=300, batch_size=10, show_metric=True)

model.save('model/dnn/face_model.rflearn')

pred = model.predict(x_test)
# print(pred)
# print(type(pred))
for _i, _p in enumerate(pred):
	_max_sort = np.argsort(-_p)
	# print(_max_sort)
	_max = _max_sort[0]
	# print(_i, _max)
	real_sort = np.argsort(-np.array(y_test[_i]))
	_real_max = real_sort[0]
	print('predict: index: %s user_name: %s; real user_name: %s' %
		  (_i, l.get(_max, ''), l.get(_real_max, '')))