# 自动玩微信小游戏跳一跳

### 环境依赖

- Python
- Opencv3
- Tensorflow (如果运行`nn_play.py`)

#### 对于安卓系统
- Adb工具
- 安卓手机

#### IOS系统 (参考[这里](https://testerhome.com/topics/7220)进行安装)
- iPhone
- Mac
- WebDriverAgent
- facebook-wda
- imobiledevice

### 定位算法
- Multiscale-search
- CV based fast-search
- Convolutional Neural Network based coarse-to-fine model

想要了解算法细节，请参见[https://zhuanlan.zhihu.com/p/32636329](https://zhuanlan.zhihu.com/p/32636329).

**注意：CV based fast-search现在只能支持Android系统**

### 运行

如果你用的是Android手机，你可以运行

	python play.py --phone Android --sensitivity 2.045

如果你用的是iPhone，需要下载训练好的模型，然后运行

	python nn_play.py --phone IOS --sensitivity 2.045

- `--phone` 有两个选项: Android或者IOS.
- `--sensitivity` 是一个控制按压时间的系数.
- `play.py` 采用了Multiscale search和Fast search, 支持Android和IOS，但是IOS更推荐用下者
- `nn_play.py` 采用了CNN based coarse-to-fine模型，支持IOS和Android

### 性能

我们的算法可以正确地检测出小人（绿色）和目标（红色）位置。

用这份代码非常容易刷榜，但是我在玩了运行了一个半小时之后，在859跳时选择狗带。

<div align="center">
<img align="center" src="resource/state_859.png" width="250" alt="state_859">
<img align="center" src="resource/state_859_res.png" width="250" alt="state_859">
<img align="center" src="resource/sota.png" width="250" alt="sota">
</div>
<br/>

### 样例视频

下面有一份样例视频，excited！

[![微信跳一跳](https://img.youtube.com/vi/OeTI2Kx8Ehc/0.jpg)](https://youtu.be/OeTI2Kx8Ehc "自动玩微信小游戏跳一跳")

### 训好的模型以及训练数据

训练好的CNN模型和训练数据可以从下面的链接下载
- [Baidu Drive](https://pan.baidu.com/s/1c2rrlra)
- [Google Drive](https://drive.google.com/drive/folders/1tCUf2krzMpkQh_RJL02x0z__4j7MaUI4?usp=sharing)

**如果你想从头自己训练：** 下载好数据到任意目录下，然后修改`cnn_coarse_to_fine/data_provider`目录下所有文件的`self.data_dir`路径。

**如果你只想跑一下我们的模型:** 下载好train log文件（包括`train_logs_coarse` and `train_log_fine`）并解压到`resource`目录。

