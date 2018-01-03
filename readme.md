# 自动玩微信小游戏跳一跳

### Requirements

- Python
- Opencv
- Adb tools
- Android Phone

### Install

```
pip3 install -r requirements.txt
```

### Run

	python play.py --phone Android --sensitivity 2.045

- `--phone` has two options: Android or IOS.
- `--sensitivity` is the constant parameter that controls the pressing time.


Our method can correctly detect the positions of the man (green dot) and the destination (red dot).

It is easy to reach the state of art as long as you like.
But I choose to go die after 859 jumps for about 1.5 hours.

<div align="center">
<img align="center" src="resource/state_859.png" width="250" alt="state_859">
<img align="center" src="resource/state_859_res.png" width="250" alt="state_859">
<img align="center" src="resource/sota.png" width="250" alt="sota">
</div>
<br/>

### Demo Video

Here is a video demo (old version). Current version is much faster than this one. Excited!

[![微信跳一跳](https://img.youtube.com/vi/MQ0SCnOcjaI/0.jpg)](https://youtu.be/MQ0SCnOcjaI "自动玩微信小游戏跳一跳")

Special thanks to [An](https://github.com/Richard-An)'s hacking ideas for speeding up.

