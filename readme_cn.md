# 自动玩微信小游戏跳一跳

## 环境依赖

- Python
- Opencv3
- Tensorflow

#### 对于安卓系统

- Adb工具
- 安卓手机

#### IOS系统 (参考[这里](https://testerhome.com/topics/7220)进行安装)

- iPhone
- Mac
- WebDriverAgent
- facebook-wda
- imobiledevice

## 定位算法

- Multiscale search
- Fast search
- CNN-based coarse-to-fine model

想要了解算法细节，请参见[https://zhuanlan.zhihu.com/p/32636329](https://zhuanlan.zhihu.com/p/32636329).

**注意：CV based fast-search现在只能支持Android系统**

## 运行

在你运行我们的代码之前，请用USB连接好你的手机。

如果是Android手机，在开发者选项里打开USB调试模式，在终端输入`adb devices`，确保设备列表不为空。
如果是iPhone手机，确保你有一台mac电脑，然后照着这个[连接](https://testerhome.com/topics/7220)去做准备工作。

**强烈推荐**下载预训练好的模型（参考后面给出的链接）并且运行下面的代码

	python nn_play.py --phone Android --sensitivity 2.045

当然你也可以使用`play.py`，只要运行

	python play.py --phone Android --sensitivity 2.045

- `--phone` 有两个选项: Android或者IOS.
- `--sensitivity` 是一个控制按压时间的系数.
- `nn_play.py` 采用了CNN-based coarse-to-fine模型，支持Android和IOS（鲁棒性更好，适用性强）
- `play.py` 采用了Multiscale search和Fast search算法, 支持Android和IOS（有的时候在其他手机下效果会差）

## 性能

我们的算法可以正确地检测出小人（绿色）和目标（红色）位置。

用这份代码非常容易刷榜，但是我在玩了运行了一个半小时之后，在859跳时选择狗带。

<div align="center">
<img align="center" src="resource/state_859.png" width="250" alt="state_859">
<img align="center" src="resource/state_859_res.png" width="250" alt="state_859">
<img align="center" src="resource/sota.png" width="250" alt="sota">
</div>
<br/>

## 样例视频

下面有一份样例视频，excited！

[![微信跳一跳](https://img.youtube.com/vi/OeTI2Kx8Ehc/0.jpg)](https://youtu.be/OeTI2Kx8Ehc "自动玩微信小游戏跳一跳")

## 训好的模型以及训练数据

训练好的CNN模型和训练数据可以从下面的链接下载
- [Baidu Drive](https://pan.baidu.com/s/1c2rrlra)
- [Google Drive](https://drive.google.com/drive/folders/1tCUf2krzMpkQh_RJL02x0z__4j7MaUI4?usp=sharing)

**如果你想从头自己训练：** 下载好数据到任意目录下，然后修改`cnn_coarse_to_fine/data_provider`目录下所有文件的`self.data_dir`路径。

**如果你只想跑一下我们的模型:** 下载好train log文件（包括`train_logs_coarse` and `train_logs_fine`）并解压到`resource`目录。

## 如何自己训练CNN模型？

0. 按照上述步骤下载并解压训练数据，并修改 `cnn_coarse_to_fine/data_provider` 文件夹下面的所有python文件的`self.data_dir` 选项到数据所在的路径。
0. `base.large` 是coarse model的模型文件夹 `base.fine` 是fine model的模型文件夹, 其他在 `cnn_coarse_to_fine/config` 文件夹下面的模型我们都没有使用，但是如果你感兴趣，你可以训练这些模型，或者训练自己构建的模型。
0. 运行 `python3 train.py -g 0` 训练模型，`-g`指定使用的GPU，如果你没有GPU，训练模型是不推荐的，因为使用CPU训练模型速度过于缓慢。
0. 模型训练好之后，复制或移动 `.ckpt` 文件到训练日志文件夹(`train_logs_coarse` 和 `train_logs_fine`) 来使用训练好的模型。

