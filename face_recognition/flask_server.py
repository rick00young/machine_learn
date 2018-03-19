from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

from face_predict import FaceRecognition

face_predict = FaceRecognition()


@app.route('/')
def index():
	return render_template('client.html')


@app.route('/camera')
def camera():
	return render_template('camera.html')


@socketio.on('camera', namespace='/test')
def test_message(message):
	print(message)
	img_data = message.get('data', '')
	img_data = img_data.split(',')
	img_data = img_data[1] if len(img_data) > 0 else ''
	output = ''
	if img_data:
		output = face_predict.predict(camera_data=img_data)
	emit('predict', {'data': output})


@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
	emit('my response', {'data': message['data']}, broadcast=True)


@socketio.on('connect', namespace='/test')
def test_connect():
	emit('my response', {'data': 'Connected'})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
	print('Client disconnected')


'''
from PIL import Image
import cv2
    # Take in base64 string and return cv image
    def stringToRGB(base64_string):
        imgdata = base64.b64decode(str(base64_string))
        image = Image.open(io.BytesIO(imgdata))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
'''

if __name__ == '__main__':
	socketio.run(app, debug=True)
