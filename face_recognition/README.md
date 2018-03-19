## 基于opencv3, dlib, tensorflow的人脸识别
本项目支持静态图片人脸识别,摄像头实时人脸识别.

支持的神经网络模型有: 深度学习模型和卷积学习模型(待补充)

#### 所使用的图片目录结构
建议图片越多越好.
```
├── faces_img
│   ├── huang_bo
│   │   └── u=3102512890,3975914519&fm=27&gp=0.jpg
│   ├── huang_xiao_ming
│   │   ├── 003j79o0zy6VFX6k8xn55&690.jpg
│   │   └── u=2946267319,1491030668&fm=27&gp=0.jpg
│   ├── katty_perrry
│   │   ├── 498f6ad4-484d-44aa-acbf-f57df983608f_x300.jpg
│   │   └── u=1916172799,2107868869&fm=27&gp=0.jpg
│   ├── sun_li
│   │   ├── 05-174558_473.jpg
│   │   ├── 20096215114.18880119.jpg
│   │   ├── ntk35109.jpg
│   │   └── yule0010.jpg
│   ├── tayler_swift
│   │   ├── 1.jpg
│   │   ├── 115002.75540645_500.jpg
│   │   ├── 12833197222565.jpg
│   │   └── 2.jpg
│   ├── tong_li_ya
│   │   ├── download\ (2).jpg
│   │   ├── download.jpg
│   │   ├── u=2378236490,1588767966&fm=27&gp=0.jpg
│   │   ├── u=44453220,1966602141&fm=11&gp=0.jpg
│   ├── zhao_li_ying
│   │   ├── download\ (2).jpg
│   │   ├── u=2121715364,858729967&fm=27&gp=0.jpg
│   │   └── u=511209731,3971680570&fm=27&gp=0.jpg
│   └── zhou_xun
│       ├── u=1599197600,416336198&fm=27&gp=0.jpg
│       └── u=707313968,154684346&fm=27&gp=0.jpg


bpy3 camera_face_recognition.py -p ~/src/dlib/model/shape_predictor_68_face_landmarks.dat -c ~/src/dlib/model/dlib_face_recognition_resnet_model_v1.dat

```


运用html5的websocket以及flask_socketio,可以通过浏览器实时人脸识别.

## 依赖
```
//python
pip install fask flask_socketio dlib opencv3

//javascipt
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.0.4/socket.io.js"></script>

另外需要去dlib官司网下载两个模型:
shape_predictor_68_face_landmarks.dat
dlib_face_recognition_resnet_model_v1.dat
来替换 face_recognition.py 里面相应的路径
```

支持的神经网络模型有: 深度学习模型和卷积学习模型(待补充)

### demo运行及测试

```
├── README.md
├── __init__.py
├── camera_face_recognition.py
├── create_face_vector.py
├── data
│   └── feature.pickle
├── demo.jpg
├── dnn.py
├── face_predict.py
├── face_test.py
├── flask_server.py
├── load_face_feature.py
├── model
│   ├── cnn
│   └── dnn
│       ├── checkpoint
│       ├── face_model.rflearn.data-00000-of-00001
│       ├── face_model.rflearn.index
│       └── face_model.rflearn.meta
├── predict.py
└── templates
    ├── camera.html
    └── client.html

```

##### 运行步骤
1. python3 flask_server.py
2. chrome或firefox里输入http://127.0.0.1:5000/camera
3. 点击同意调用摄像头协议即可看到结果

##### 注意
可以用一些李冰冰,周讯,黄晓明等的图片进行测试,此demo用的dnn的训练模型,cnn有待补充.


#### 结果展示:
![实时人脸识别结果](https://github.com/rick00young/machine_learn/blob/master/face_recognition/demo.jpg "实时人脸识别结果")