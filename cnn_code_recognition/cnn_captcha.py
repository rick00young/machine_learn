import tensorflow as tf
import math

from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
import string


class generateCaptcha():
	def __init__(self,
	             width=160,  # 验证码图片的宽
	             height=60,  # 验证码图片的高
	             char_num=4,  # 验证码字符个数
	             characters=string.digits + string.ascii_uppercase + string.ascii_lowercase):  # 验证码组成，数字+大写字母+小写字母
		self.width = width
		self.height = height
		self.char_num = char_num
		self.characters = characters
		self.classes = len(characters)

	def gen_captcha(self, batch_size=50):
		X = np.zeros([batch_size, self.height, self.width, 1])
		img = np.zeros((self.height, self.width), dtype=np.uint8)
		Y = np.zeros([batch_size, self.char_num, self.classes])
		image = ImageCaptcha(width=self.width, height=self.height)

		while True:
			for i in range(batch_size):
				captcha_str = ''.join(random.sample(self.characters, self.char_num))
				img = image.generate_image(captcha_str).convert('L')
				img = np.array(img.getdata())
				X[i] = np.reshape(img, [self.height, self.width, 1]) / 255.0
				for j, ch in enumerate(captcha_str):
					Y[i, j, self.characters.find(ch)] = 1
			Y = np.reshape(Y, (batch_size, self.char_num * self.classes))
			yield X, Y

	def decode_captcha(self, y):
		y = np.reshape(y, (len(y), self.char_num, self.classes))
		return ''.join(self.characters[x] for x in np.argmax(y, axis=2)[0, :])

	def get_parameter(self):
		return self.width, self.height, self.char_num, self.characters, self.classes

	def gen_test_captcha(self):
		image = ImageCaptcha(width=self.width, height=self.height)
		captcha_str = ''.join(random.sample(self.characters, self.char_num))
		img = image.generate_image(captcha_str)
		img.save(captcha_str + '.jpg')


class captchaModel():
	def __init__(self,
	             width=160,
	             height=60,
	             char_num=4,
	             classes=62):
		self.width = width
		self.height = height
		self.char_num = char_num
		self.classes = classes

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
		                      strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def create_model(self, x_images, keep_prob):
		# first layer
		w_conv1 = self.weight_variable([5, 5, 1, 32])
		b_conv1 = self.bias_variable([32])
		h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_images, w_conv1), b_conv1))
		h_pool1 = self.max_pool_2x2(h_conv1)
		h_dropout1 = tf.nn.dropout(h_pool1, keep_prob)
		conv_width = math.ceil(self.width / 2)
		conv_height = math.ceil(self.height / 2)

		# second layer
		w_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])
		h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout1, w_conv2), b_conv2))
		h_pool2 = self.max_pool_2x2(h_conv2)
		h_dropout2 = tf.nn.dropout(h_pool2, keep_prob)
		conv_width = math.ceil(conv_width / 2)
		conv_height = math.ceil(conv_height / 2)

		# third layer
		w_conv3 = self.weight_variable([5, 5, 64, 64])
		b_conv3 = self.bias_variable([64])
		h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout2, w_conv3), b_conv3))
		h_pool3 = self.max_pool_2x2(h_conv3)
		h_dropout3 = tf.nn.dropout(h_pool3, keep_prob)
		conv_width = math.ceil(conv_width / 2)
		conv_height = math.ceil(conv_height / 2)

		# first fully layer
		conv_width = int(conv_width)
		conv_height = int(conv_height)
		w_fc1 = self.weight_variable([64 * conv_width * conv_height, 1024])
		b_fc1 = self.bias_variable([1024])
		h_dropout3_flat = tf.reshape(h_dropout3, [-1, 64 * conv_width * conv_height])
		h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# second fully layer
		w_fc2 = self.weight_variable([1024, self.char_num * self.classes])
		b_fc2 = self.bias_variable([self.char_num * self.classes])
		y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

		return y_conv


if __name__ == '__main__':
	captcha = generateCaptcha()
	width, height, char_num, characters, classes = captcha.get_parameter()

	x = tf.placeholder(tf.float32, [None, height, width, 1])
	y_ = tf.placeholder(tf.float32, [None, char_num * classes])
	keep_prob = tf.placeholder(tf.float32)

	model = captchaModel(width, height, char_num, classes)
	y_conv = model.create_model(x, keep_prob)
	cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	predict = tf.reshape(y_conv, [-1, char_num, classes])
	real = tf.reshape(y_, [-1, char_num, classes])
	correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
	correct_prediction = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		step = 0
		while True:
			batch_x, batch_y = next(captcha.gen_captcha(64))
			_, loss, acc = sess.run([train_step, cross_entropy, accuracy],
			                        feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
			print('step:%d,loss:%f,acc:%f' % (step, loss, acc))
			if step % 100 == 0:
				batch_x_test, batch_y_test = next(captcha.gen_captcha(100))
				acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
				print('###############################################step:%d,accuracy:%f' % (step, acc))
				if acc > 0.99:
					saver.save(sess, "model/capcha_model.ckpt")
					break
			step += 1
