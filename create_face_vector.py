import sys
import os
import dlib
import glob
from skimage import io
import pickle

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

face_feature = []
face_user = {}

for user_index, user_name in enumerate(os.listdir(faces_folder_path)):
    # print(f)
    folder = faces_folder_path + '/' + user_name
    if not os.path.isdir(folder):
        continue
    print(user_index, user_name)
    # continue
    face_user[user_index] = user_name
    # Now process all the images
    for f in glob.glob(os.path.join(folder, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if 1 != len(dets):
            print('there is no face or more than one face.place check again!')
            continue

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            # print(type(face_descriptor))
            print(face_descriptor)
            
            current_face = [x for x in face_descriptor]
            current_face.append(user_index)
            face_feature.append(current_face)
            # dlib.hit_enter_to_continue()

print(face_feature)
print(face_user)

all_feature = {'feature': face_feature, 'user_label': face_user}
with open('data/feature.pickle', 'wb') as f:
    pickle.dump(all_feature, f)

