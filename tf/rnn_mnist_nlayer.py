import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt


TIME_STEPS=28
BATCH_SIZE=128
HIDDEN_UNITS1=30
HIDDEN_UNITS=10
LEARNING_RATE=0.001
EPOCH=5000

TRAIN_EXAMPLES=42000
TEST_EXAMPLES=28000

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/Users/rick/src/ml_data/raw/MNIST_data',
								   one_hot=True)

print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

#------------------------------------Generate Data-----------------------------------------------#
# generate data
# train_frame = pd.read_csv("../Mnist/train.csv")
# test_frame = pd.read_csv("../Mnist/test.csv")

# pop the labels and one-hot coding
# train_labels_frame = train_frame.pop("label")

# get values
# one-hot on labels
# X_train = train_frame.astype(np.float32).values
# y_train=pd.get_dummies(data=train_labels_frame).values
# X_test = test_frame.astype(np.float32).values
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels


#trans the shape to (batch,time_steps,input_size)
X_train=np.reshape(X_train,newshape=(-1,28,28))
X_test=np.reshape(X_test,newshape=(-1,28,28))
#print(X_train.shape)
#print(y_dummy.shape)
#print(X_test.shape)

#-----------------------------------------------------------------------------------------------------#


#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place hoder
    # inputs = tf.placeholder(np.float32, shape=(32,40,5)) # 32 是 batch_size
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,TIME_STEPS,28),name="input_placeholder")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,10),name="pred_placeholder")

    #lstm instance
    lstm_cell1=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_cell=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)

    multi_lstm=rnn.MultiRNNCell(cells=[lstm_cell1,lstm_cell])

    #initialize to zero
    init_state=multi_lstm.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)

    #dynamic rnn
    outputs,states=tf.nn.dynamic_rnn(cell=multi_lstm,inputs=X_p,initial_state=init_state,dtype=tf.float32)
    # [batch_size, max_time, cell.output_size]
    # print(outputs.shape)
    # shape [128, ,10]
    # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    h=outputs[:,-1,:]

    _state = states[-1][1]
    _output = h

    #print(h.shape)
    #--------------------------------------------------------------------------------------------#

    #---------------------------------define loss and optimizer----------------------------------#
    cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h)
    #print(loss.shape)

    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss)

    init=tf.global_variables_initializer()


#-------------------------------------------Define Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1,EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses=[]
        accus=[]
        #test_losses=[]
        print("epoch:",epoch)
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            _,train_loss,accu=sess.run(
                    fetches=(optimizer,cross_loss,accuracy),
                    feed_dict={
                            X_p:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            train_losses.append(train_loss)
            accus.append(accu)
        print('_output')
        print(_output)
        print('_state')
        print(_state)
        print('_output == _state', tf.reduce_mean(tf.cast(tf.equal(_output, _state), 'float')))
        print("average training loss:", sum(train_losses) / len(train_losses))