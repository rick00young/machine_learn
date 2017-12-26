import tflearn
import sys
import dlib
import glob
import os
from skimage import io
import numpy as np

import load_face_feature
x, y, l = load_face_feature.load_feature()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]

# win = dlib.image_window()


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def get_face_feature(face_path, face_limit=0):
	result = []
	if os.path.isdir(face_path):
		for f in glob.glob(os.path.join(face_path, "*.jpg")):
			face = parse_face_feature(img_path=f, face_limit=face_limit)
			for i in face:
				result.append(i)
	elif os.path.isfile(face_path):
		face = parse_face_feature(img_path=face_path, face_limit=face_limit)
		result.append(face)

	return result

def parse_face_feature(img_path, face_limit=0):
	print("Processing file: {}".format(img_path))
	img = io.imread(img_path)
	result = []
	# win.clear_overlay()
	# win.set_image(img)
	dets = detector(img, 1)
	print("Number of faces detected: {}".format(len(dets)))
	if 0 == len(dets):
		print('there is no face.place check again!')
		return []

	if face_limit > 0 and len(dets) > face_limit:
		raise Exception('we found faces: %s which is not match from limit: %s' % (len(dets), face_limit))

	# Now process each face we found.
	for k, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
			k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = sp(img, d)
		# Draw the face landmarks on the screen so we can see what face is currently being processed.
		# win.clear_overlay()
		# win.add_overlay(d)
		# win.add_overlay(shape)

		face_descriptor = facerec.compute_face_descriptor(img, shape)
		print(type(face_descriptor))
		face = {
			'img': img_path,
			'left': d.left(),
			'top': d.top(),
			'right': d.right(),
			'bottom': d.bottom(),
			'feature': [x for x in face_descriptor]
		}
		result.append(face)
	return result



net = tflearn.input_data(shape=[None, 128])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, len(l), activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.load('model/dnn/face_model.rflearn')

feature = get_face_feature(face_path=faces_folder_path)
real_user = faces_folder_path.split('/')[-1] if not faces_folder_path.endswith('/') else faces_folder_path.split('/')[-2:-1][0]

for i in feature:
	_face = i.get('feature', [])
	if not _face:
		continue
	predict = model.predict([_face])
	for p in predict:
		score = np.argsort(-p)
		max_index = score[0]
		print('predict the user is: %s; real_user is : %s' % (l.get(max_index, ''), real_user))


# if '__main__' == __name__:
# 	feature = get_face_feature(face_path=faces_folder_path)
# 	print(feature)
