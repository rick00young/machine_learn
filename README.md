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
```

#### 结果展示:
![实时人脸识别结果](https://github.com/rick00young/face_recognition/blob/master/demo.jpg "实时人脸识别结果")