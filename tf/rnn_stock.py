import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import random
import sys


f = open('dataset_1.csv')

df = pd.read_csv(f)

data = np.array(df['max'])
# print(data)
data = data[::-1]
plt.figure()
# plt.plot(data)
# plt.show()

normalize_data = (data - np.mean(data))/np.std(data)
print(normalize_data.shape)
normalize_data = normalize_data[:, np.newaxis]
print(normalize_data)
print(normalize_data.shape)

#———————————————————形成训练集—————————————————————
#设置常量
TIME_STEP=20      #时间步
RNN_UNIT1= 30      #hidden layer units
RNN_UNIT= 1      #hidden layer units
BATCH_SIZE=60     #每一批次训练多少个样例
INPUT_SIZE=1      #输入层维度
OUTPUT_SIZE=1     #输出层维度
LR=0.0006         #学习率
EPOCH = 160
train_x,train_y=[],[]   #训练集
test_x , test_y = [], []

for i in range(len(normalize_data) - TIME_STEP):
	x = normalize_data[i:i + TIME_STEP]
	y = normalize_data[i + TIME_STEP]
	# print(y.tolist())
	train_x.append(x.tolist())
	train_y.append(y.tolist())
	if random.randint(0, 10) > 9:
		test_x.append(x)
		test_y.append(y)

# print(train_y)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


print('train_x', train_x.shape)
print('train_y', train_y.shape)
print('text_x', test_x.shape)
# sys.exit(9)

X_p = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEP, 1),
	                     name='input_placeholder')
y_p = tf.placeholder(dtype=tf.float32, shape=(None, 1),
	                     name='pred_placeholder')


def lstm(batch):
	lstm_cell1 = rnn.BasicLSTMCell(num_units=RNN_UNIT1)
	lstm_cell = rnn.BasicLSTMCell(num_units=RNN_UNIT)
	multi_cell = rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell])

	init_state = multi_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

	outputs, states = tf.nn.dynamic_rnn(cell=multi_cell, inputs=X_p,
	                                    initial_state=init_state,
	                                    dtype=tf.float32)
	return outputs, states
	# h = outputs[:, -1, :]
	# mse = tf.losses.mean_squared_error(labels=y_p, predictions=h)
	# optimizer = tf.train.AdamOptimizer(LR).minimize(mse)
	# init = tf.global_variables_initializer()

def train_lstm():
	outputs, states = lstm(1)
	h = outputs[:, -1, :]
	mse = tf.losses.mean_squared_error(labels=y_p, predictions=h)
	optimizer = tf.train.AdamOptimizer(LR).minimize(mse)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(tf.global_variables())

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(1, EPOCH):
			results = np.zeros(shape=(len(train_x), 1))
			train_losses = []
			test_losses = []
			_train_loss = None
			print('epoch:', epoch)
			for j in range(len(train_x)//BATCH_SIZE):
				_, _train_loss = sess.run(
					fetches=(optimizer, mse),
					feed_dict={
						X_p: train_x[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
						y_p: train_y[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
					}
				)
				train_losses.append(_train_loss)
			print('average training loss:', sum(train_losses)/len(train_losses))
			print('mse:',_train_loss)
		saver.save(sess, 'model/stock.model')



def predict():
	outputs, states = lstm(1)
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		model_file = tf.train.latest_checkpoint('model')
		saver.restore(sess, model_file)
		h = outputs[:, -1, :]
		for j in range(len(test_x)//BATCH_SIZE):
			_test_h = sess.run(
				fetches=(h),
				feed_dict={
					X_p: test_x[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
					# y_p: test_y[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
				}
			)
			_y = test_y[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
			# print('_test_h', _test_h)
			print('accuracy:', sum(np.square(_test_h - _y))/BATCH_SIZE)

		# print('test_loss', _test_h)

train_lstm()
# predict()