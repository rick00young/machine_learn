# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import tflearn
import base64
from PIL import Image
import io

import load_face_feature

class FaceRecognition():
	def __init__(self):
		x, y, l = load_face_feature.load_feature()
		self.x = x
		self.y = y
		self.l = l

		shape_predictor = '/Users/rick/src/dlib/model/shape_predictor_68_face_landmarks.dat'
		rec_model = '/Users/rick/src/dlib/model/dlib_face_recognition_resnet_model_v1.dat'
		print("[INFO] loading facial landmark predictor...")
		self.detector = dlib.get_frontal_face_detector()
		# shape_predictor = dlib.shape_predictor(args["shape_predictor"])
		# facerec = dlib.face_recognition_model_v1(args['rec_model'])
		self.shape_predictor = dlib.shape_predictor(shape_predictor)
		self.facerec = dlib.face_recognition_model_v1(rec_model)

		net = tflearn.input_data(shape=[None, 128])
		net = tflearn.fully_connected(net, 32)
		net = tflearn.fully_connected(net, 32)
		net = tflearn.fully_connected(net, len(self.l), activation='softmax')
		net = tflearn.regression(net)

		# Define model
		model = tflearn.DNN(net)
		# Start training (apply gradient descent algorithm)
		model.load('model/dnn/face_model.rflearn')
		self.model = model


	def get_predict_user_name(self, predict, face_user={}):
		result = []
		for i in predict:
			pre = np.argsort(-np.array(i))
			max_index = pre[0]
			result.append(face_user.get(max_index, 'Unknown'))
		return result

	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--shape-predictor", default=True,
	# 				help="path to facial landmark predictor")
	# ap.add_argument("-ocr", "--picamera", type=int, default=-1,
	# 				help="whether or not the Raspberry Pi camera should be used")
	# ap.add_argument('-c', '--rec-model', required=True,
	# 				help='reco model to parse face feature!')
	# args = vars(ap.parse_args())
	#
	def stringToRGB(self, base64_string):
		imgdata = base64.b64decode(str(base64_string))
		image = Image.open(io.BytesIO(imgdata))
		return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

	def predict(self, camera_data=None):
		# print("[INFO] camera sensor warming up...")
		# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
		# time.sleep(1.0)
		# cap = cv2.VideoCapture(0)

		# loop over the frames from the video stream
		if not camera_data:
			print('there is no camera data')
			return ''
		frame = self.stringToRGB(camera_data)
		# while True:
		# ret, frame = cap.read()

		# print(frame)
		# print(ret)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# _gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
		#
		# # detect faces in the grayscale frame
		# _gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
		_gray = frame
		rects = self.detector(frame, 0)
		# print(frame)

		# loop over the face detections
		# for rect in rects:

		for k, d in enumerate(rects):

			shape = self.shape_predictor(_gray, d)
			_shape = face_utils.shape_to_np(shape)

			for (x, y) in _shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			face_descriptor = self.facerec.compute_face_descriptor(_gray, shape)
			# print(face_descriptor)
			predict = self.model.predict([face_descriptor])
			print(predict)
			user = self.get_predict_user_name(predict=predict, face_user=self.l)
			print(user)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame, user[0], (d.left()+20, d.top()+30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
			# cv2.rectangle(frame, (384, 0), (510, 128), (0, 255, 0), 3)
			cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)


		# show the frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

		# if key == ord("q"):
		# 	break

		# cv2.waitKey()

		# do a bit of cleanup
		# cap.release()
		# cv2.destroyAllWindows()
		image = Image.fromarray(frame)
		image.resize((360, 270))
		output_buffer = io.BytesIO()
		image.save(output_buffer, format='JPEG')
		byte_data = output_buffer.getvalue()
		base64_str = base64.b64encode(byte_data)
		# image.show()
		return '%s,%s' % ('data:image/jpeg;base64', str(base64_str, encoding='utf-8'))

		# print(base64_str)
		# print(base64.b64encode(image.tobytes()))
		# print()














