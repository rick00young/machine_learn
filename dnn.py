import tflearn
import sys, os
from sklearn.cross_validation import train_test_split

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
model.fit(x_train, y_train, n_epoch=100, batch_size=4, show_metric=True)

model.save('model/dnn/')