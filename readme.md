# 自动玩微信小游戏跳一跳

中文说明请点[这里](https://github.com/Prinsphield/Wechat_AutoJump/blob/master/readme_cn.md)

## Requirements

- Python
- Opencv3
- Tensorflow

#### for Android

- Adb tools
- Android Phone

#### for IOS (Refer to this [site](https://testerhome.com/topics/7220) for installation)

- iPhone
- Mac
- WebDriverAgent
- facebook-wda
- imobiledevice

## Algorithms for Localization

- Multiscale search
- Fast search
- CNN-based coarse-to-fine model

For algorithm details, please go to [https://zhuanlan.zhihu.com/p/32636329](https://zhuanlan.zhihu.com/p/32636329).

**Notice: CV based fast-search only support Android for now**

## Run

Before running our code, connect to your phone via USB.

If Android phone, open the USB debugging at developer options enter `adb devices` to ensure that the list is not empty.
If iPhone, please ensure that you have a mac. Then following this [link](https://testerhome.com/topics/7220) for preparation.

It is **recommended** to download the pre-trained model following the link below and run the following code

	python nn_play.py --phone Android --sensitivity 2.045

You can also try `play.py` by running the following code

	python play.py --phone Android --sensitivity 2.045

- `--phone` has two options: Android or IOS.
- `--sensitivity` is the constant parameter that controls the pressing time.
- `nn_play.py` uses CNN-based coarse-to-fine model, supporting Android and IOS (more robust)
- `play.py` uses multiscale search and fast search algorithms, supporting Android and IOS (it may fail sometimes in other phones)

## Performance

Our method can correctly detect the positions of the man (green dot) and the destination (red dot).

It is easy to reach the state of art as long as you like.
But I choose to go die after 859 jumps for about 1.5 hours.

<div align="center">
<img align="center" src="resource/state_859.png" width="250" alt="state_859">
<img align="center" src="resource/state_859_res.png" width="250" alt="state_859">
<img align="center" src="resource/sota.png" width="250" alt="sota">
</div>
<br/>

## Demo Video

Here is a video demo. Excited!

[![微信跳一跳](https://img.youtube.com/vi/OeTI2Kx8Ehc/0.jpg)](https://youtu.be/OeTI2Kx8Ehc "自动玩微信小游戏跳一跳")

## Train Log & Data

CNN train log and train&validation data avaliable at
- [Baidu Drive](https://pan.baidu.com/s/1c2rrlra)
- [Google Drive](https://drive.google.com/drive/folders/1tCUf2krzMpkQh_RJL02x0z__4j7MaUI4?usp=sharing)

**Training:** download and untar data into any directory, and then modify `self.data_dir` in those files under `cnn_coarse_to_fine/data_provider` directory.

**Inference:** download and unzip train log dirs(`train_logs_coarse` and `train_logs_fine`) into `resource` directory.

## How to Train CNN models by yourself?

0. Download and untar data into any directory, and then modify `self.data_dir` in those files under `cnn_coarse_to_fine/data_provider` directory.
0. `base.large` is model dir for coarse model, `base.fine` is model dir for fine model, other dirs under `cnn_coarse_to_fine/config` are models we don't use, but if you have interests, you can try train other models by yourself.
0. Run `python3 train.py -g 0` to train your model, `-g` to specify GPU to use, if you don't have GPU, training model is not recommended because training speed with CPU is very slow.
0. After training, move or copy `.ckpt` file to train log dirs(`train_logs_coarse` and `train_logs_fine`) for use.

