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

import load_face_feature
x, y, l = load_face_feature.load_feature()



def get_predict_user_name(predict, face_user={}):
	result = []
	for i in predict:
		pre = np.argsort(-np.array(i))
		max_index = pre[0]
		result.append(face_user.get(max_index, 'Unknown'))
	return result

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
				help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
				help="whether or not the Raspberry Pi camera should be used")
ap.add_argument('-c', '--rec-model', required=True,
				help='reco model to parse face feature!')
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(args["shape_predictor"])
facerec = dlib.face_recognition_model_v1(args['rec_model'])

# print("[INFO] camera sensor warming up...")
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
# time.sleep(1.0)
cap = cv2.VideoCapture(0)

net = tflearn.input_data(shape=[None, 128])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, len(l), activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.load('model/dnn/face_model.rflearn')

# loop over the frames from the video stream
while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	#
	# # detect faces in the grayscale frame
	rects = detector(gray, 0)
	# print(frame)

	# loop over the face detections
	# for rect in rects:

	for k, d in enumerate(rects):

		shape = shape_predictor(_gray, d)
		_shape = face_utils.shape_to_np(shape)

		for (x, y) in _shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		face_descriptor = facerec.compute_face_descriptor(_gray, shape)
		# print(face_descriptor)
		predict = model.predict([face_descriptor])
		print(predict)
		user = get_predict_user_name(predict=predict, face_user=l)
		print(user)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, user[0], (d.left()+20, d.top()+30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
		# cv2.rectangle(frame, (384, 0), (510, 128), (0, 255, 0), 3)
		cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break



# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()














