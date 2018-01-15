# Deep Song Classification

深度学习对歌曲分类:
> 通过音频处理工具sox,对音频进行特征提取,并生成特征频谱图,根据特征频谱图,对音频进行深度学习并分类,本人共训练了300多首歌曲,歌曲分别来酷我,QQ音乐,虾米和网易云音乐.通过各大音乐APP,根据其标签下裁相应的歌曲.

> 本分类准确率可以达到76%.效果不是很好,也许跟原数据的分类本身就存在错误有关.

依赖库

```
eyed3
sox --with-lame
tensorflow
tflearn
```

- 创建目录 Data/Raw/
- 将音乐放置到上面的目录

对歌曲进行切片

```
python main.py slice
```

训练分类器:

```
python main.py train
```

分类器测试:

```
python main.py test
```


参考资料:[这里](https://github.com/despoisj/DeepAudioClassification)