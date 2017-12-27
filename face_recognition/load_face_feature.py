import pickle


def load_feature():
	all_feature = pickle.load(open('data/feature.pickle', 'rb'))

	face_feature = all_feature.get('feature', [])
	face_user = all_feature.get('user_label', {})

	user_num = len(face_user)

	X = []
	Y = []

	for i in face_feature:
		X.append(i[0:-1])
		label_i = i[-1]
		Y.append([1 if label_i == l else 0 for l in range(0, user_num)])

	return X, Y, face_user

if '__main__' == __name__:
	x, y, l = load_feature()
	print(y)
	print(l)
	print(len(x[1]))