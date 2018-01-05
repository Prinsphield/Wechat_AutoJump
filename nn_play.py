# -*- coding:utf-8 -*-
# Created Time: 六 12/30 13:49:21 2017
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import time
import os, glob, shutil
import cv2
import argparse
import tensorflow as tf
from model import JumpModel
from model_fine import JumpModelFine
import random

def multi_scale_search(pivot, screen, range=0.3, num=10):
    H, W = screen.shape[:2]
    h, w = pivot.shape[:2]

    found = None
    for scale in np.linspace(1-range, 1+range, num)[::-1]:
        resized = cv2.resize(screen, (int(W * scale), int(H * scale)))
        r = W / float(resized.shape[1])
        if resized.shape[0] < h or resized.shape[1] < w:
            break
        res = cv2.matchTemplate(resized, pivot, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= res.max())
        pos_h, pos_w = list(zip(*loc))[0]

        if found is None or res.max() > found[-1]:
            found = (pos_h, pos_w, r, res.max())

    if found is None: return (0,0,0,0,0)
    pos_h, pos_w, r, score = found
    start_h, start_w = int(pos_h * r), int(pos_w * r)
    end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
    return [start_h, start_w, end_h, end_w, score]

class WechatAutoJump(object):
    def __init__(self, phone, sensitivity, debug, resource_dir):
        self.phone = phone
        self.sensitivity = sensitivity
        self.debug = debug
        self.resource_dir = resource_dir
        self.step = 0
        self.ckpt = os.path.join(self.resource_dir, 'train_logs_coarse/best_model.ckpt-13999')
        self.ckpt_fine = os.path.join(self.resource_dir, 'train_log_fine/best_model.ckpt-53999')
        self.load_resource()
        if self.phone == 'IOS':
            import wda
            self.client = wda.Client('http://localhost:8100')
            self.s = self.client.session()
        if self.debug:
            if not os.path.exists(self.debug):
                os.mkdir(self.debug)

    def load_resource(self):
        self.player = cv2.imread(os.path.join(self.resource_dir, 'player.png'), 0)
        # network initization
        self.net = JumpModel()
        self.net_fine = JumpModelFine()
        self.img = tf.placeholder(tf.float32, [None, 640, 720, 3], name='img')
        self.img_fine = tf.placeholder(tf.float32, [None, 320, 320, 3], name='img_fine')
        self.label = tf.placeholder(tf.float32, [None, 2], name='label')
        self.is_training = tf.placeholder(np.bool, name='is_training')
        self.keep_prob = tf.placeholder(np.float32, name='keep_prob')
        self.pred = self.net.forward(self.img, self.is_training, self.keep_prob)
        self.pred_fine = self.net_fine.forward(self.img_fine, self.is_training, self.keep_prob)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.all_variables()
        var_coarse = [k for k in all_vars if k.name.startswith('coarse')]
        var_fine = [k for k in all_vars if k.name.startswith('fine')]
        self.saver_coarse = tf.train.Saver(var_coarse)
        self.saver_fine = tf.train.Saver(var_fine)
        self.saver_coarse.restore(self.sess, self.ckpt)
        self.saver_fine.restore(self.sess, self.ckpt_fine)
        print('==== successfully restored ====')

    def get_current_state(self):
        if self.phone == 'Android':
            os.system('adb shell screencap -p /sdcard/1.png')
            os.system('adb pull /sdcard/1.png state.png')
        elif self.phone == 'IOS':
            self.client.screenshot('state.png')

        if self.debug:
            shutil.copyfile('state.png', os.path.join(self.debug, 'state_{:03d}.png'.format(self.step)))

        state = cv2.imread('state.png')
        self.resolution = state.shape[:2]
        scale = state.shape[1] / 720.
        state = cv2.resize(state, (720, int(state.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
        if state.shape[0] > 1280:
            s = state.shape[0] - 1280
            state = state[s:,:,:]
        elif state.shape[0] < 1280:
            s = 1280 - state.shape[0]
            state = np.concatenate((255 * np.ones((s, 720, 3), dtype=np.uint8), state), 0)
        return state

    def get_player_position(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        pos = multi_scale_search(self.player, state, 0.3, 10)
        h, w = int((pos[0] + 13 * pos[2])/14.), (pos[1] + pos[3])//2
        return np.array([h, w])

    def get_target_position(self, state, player_pos):
        feed_dict = {
            self.img: np.expand_dims(state[320:-320], 0),
            self.is_training: False,
            self.keep_prob: 1.0,
        }
        pred_out = self.sess.run(self.pred, feed_dict=feed_dict)
        pred_out = pred_out[0].astype(int)
        x1 = pred_out[0] - 160
        x2 = pred_out[0] + 160
        y1 = pred_out[1] - 160
        y2 = pred_out[1] + 160
        if y1 < 0:
            y1 = 0
            y2 = 320
        if y2 > state.shape[1]:
            y2 = state.shape[1]
            y1 = y2 - 320
        img_fine_in = state[x1: x2, y1: y2, :]
        feed_dict_fine = {
            self.img_fine: np.expand_dims(img_fine_in, 0),
            self.is_training: False,
            self.keep_prob: 1.0,
        }
        pred_out_fine = self.sess.run(self.pred_fine, feed_dict=feed_dict_fine)
        pred_out_fine = pred_out_fine[0].astype(int)
        out = pred_out_fine + np.array([x1, y1])
        return out

    def get_target_position_fast(self, state, player_pos):
        state_cut = state[:player_pos[0],:,:]
        m1 = (state_cut[:, :, 0] == 245)
        m2 = (state_cut[:, :, 1] == 245)
        m3 = (state_cut[:, :, 2] == 245)
        m = np.uint8(np.float32(m1 * m2 * m3) * 255)
        b1, b2 = cv2.connectedComponents(m)
        for i in range(1, np.max(b2) + 1):
            x, y = np.where(b2 == i)
            if len(x) > 280 and len(x) < 310:
                r_x, r_y = x, y
        h, w = int(r_x.mean()), int(r_y.mean())
        return np.array([h, w])

    def jump(self, player_pos, target_pos):
        distance = np.linalg.norm(player_pos - target_pos)
        press_time = distance * self.sensitivity
        press_time = int(press_time)
        if self.phone == 'Android':
            press_h, press_w = random.randint(300,1500), random.randint(200,800)
            cmd = 'adb shell input swipe {} {} {} {} {}'.format(press_w, press_h, press_w, press_h, press_time)
            print(cmd)
            os.system(cmd)
        elif self.phone == 'IOS':
            self.s.tap_hold(random.randint(300,1500), random.randint(200,800), press_time / 1000.)

    def debugging(self):
        current_state = self.state.copy()
        cv2.circle(current_state, (self.player_pos[1], self.player_pos[0]), 5, (0,255,0), -1)
        cv2.circle(current_state, (self.target_pos[1], self.target_pos[0]), 5, (0,0,255), -1)
        cv2.imwrite(os.path.join(self.debug, 'state_{:03d}_res_h_{}_w_{}.png'.format(self.step, self.target_pos[0], self.target_pos[1])), current_state)

    def play(self):
        self.state = self.get_current_state()
        self.player_pos = self.get_player_position(self.state)
        if self.phone == 'IOS':
            self.target_pos = self.get_target_position(self.state, self.player_pos)
            print('CNN-search: %04d' % self.step)
        else:
            try:
                self.target_pos = self.get_target_position_fast(self.state, self.player_pos)
                print('fast-search: %04d' % self.step)
            except:
                self.target_pos = self.get_target_position(self.state, self.player_pos)
                print('CNN-search: %04d' % self.step)
        if self.debug:
            self.debugging()
        self.jump(self.player_pos, self.target_pos)
        self.step += 1
        time.sleep(1.5)

    def run(self):
        try:
            while True:
                self.play()
        except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone', default='Android', choices=['Android', 'IOS'], type=str, help='mobile phone OS')
    parser.add_argument('--sensitivity', default=2.051, type=float, help='constant for press time')
    parser.add_argument('--resource', default='resource', type=str, help='resource dir')
    parser.add_argument('--debug', default=None, type=str, help='debug mode, specify a directory for storing log files.')
    args = parser.parse_args()
    # print(args)

    AI = WechatAutoJump(args.phone, args.sensitivity, args.debug, args.resource)
    AI.run()
