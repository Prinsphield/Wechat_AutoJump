# -*- coding:utf-8 -*-
# Created Time: 六 12/30 13:49:21 2017
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import time
import os, glob, shutil
import cv2
import argparse
from multiprocessing import Pool
from functools import partial
from itertools import repeat

def multi_scale_search(pivot, screen):
    H, W = screen.shape[:2]
    h, w = pivot.shape[:2]

    found = None
    for scale in np.linspace(0.7, 1.3, 10)[::-1]:
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
    def __init__(self, phone, resolution, sensitivity, debug, resource_dir):
        self.phone = phone
        self.resolution = resolution
        self.scale = self.resolution[1]/720.
        self.sensitivity = sensitivity
        self.debug = debug
        self.resource_dir = resource_dir
        self.step = 0
        self.load_resource()
        if self.phone == 'IOS':
            self.client = wda.Client()
            self.sess = self.client.session()

    def load_resource(self):
        self.player = cv2.imread(os.path.join(self.resource_dir, 'player.png'), 0)
        circle_file = glob.glob(os.path.join(self.resource_dir, 'circle/*.png'))
        table_file  = glob.glob(os.path.join(self.resource_dir, 'table/*.png'))
        self.jump_file = [cv2.imread(name, 0) for name in circle_file + table_file]

    def get_current_state(self):
        if self.phone == 'Android':  
            os.system('adb shell screencap -p /sdcard/1.png')
            os.system('adb pull /sdcard/state.png state.png')
        elif self.phone == 'IOS':
            self.client.screenshot('state.png')

        if self.debug:
            shutil.copyfile('state.png', 'state_{:03d}.png'.format(self.step))

        state = cv2.imread('state.png', 0)
        state = cv2.resize(state, (720, int(self.resolution[0] / self.scale)))
        return state

    def get_player_position(self, state):
        return multi_scale_search(self.player, state)

    def get_target_position_fast(self, state, player_pos):
        state_cut = state[:player_pos[2],:].copy()
        jump_file = self.jump_file.copy()
        pool = Pool(5)
        partial_search = partial(multi_scale_search, screen=state_cut)
        positions = pool.map(partial_search, jump_file)
        # positions = pool.starmap(multi_scale_search, zip(jump_file, repeat(state_cut)))
        pool.close()
        pool.join()
        max_ind = np.argmax(np.array(positions)[:,-1])
        target_pos = positions[max_ind]
        return target_pos

    def get_target_position(self, state, player_pos):
        state_cut = state[:player_pos[2],:]
        target_pos = None
        for target in self.jump_file:
            pos = multi_scale_search(target, state_cut)
            if target_pos is None or pos[-1] > target_pos[-1]:
                target_pos = pos
        return target_pos

    def jump(self, player_pos, target_pos):
        p_s = np.array([player_pos[2], (player_pos[1]+player_pos[3])//2])
        p_e = np.array([(target_pos[0]+target_pos[2])//2, (target_pos[1]+target_pos[3])//2])
        distance = np.linalg.norm(p_s - p_e)

        press_time = distance * self.sensitivity
        press_time = int(press_time)
        if self.phone == 'Android':
            cmd = 'adb shell input swipe 320 410 320 410 ' + str(press_time)
            print(cmd)
            os.system(cmd)
        elif self.phone == 'IOS':
            self.sess.tap_hold(200, 200, press_time)


    def debugging(self):
        current_state = self.state.copy()
        cv2.rectangle(current_state, (self.player_pos[1], self.player_pos[0]), (self.player_pos[3], self.player_pos[2]), (0,255,0), 2)
        cv2.rectangle(current_state, (self.target_pos[1], self.target_pos[0]), (self.target_pos[3], self.target_pos[2]), (0,0,255), 2)
        cv2.imwrite('state_{:03d}_res.png'.format(self.step), current_state)

    def play(self):
        self.state = self.get_current_state()
        self.player_pos = self.get_player_position(self.state)
        self.target_pos = self.get_target_position(self.state, self.player_pos)
        if self.debug:
            self.debugging()
        self.jump(self.player_pos, self.target_pos)
        self.step += 1
        time.sleep(1)

    def run(self):
        try:
            while True:
                self.play()
        except KeyboardInterrupt:
                pass

    def test_detection(self, file):
        self.state = cv2.imread(file, 0)
        self.player_pos = self.get_player_position(self.state)
        self.target_pos = self.get_target_position(self.state, self.player_pos)
        current_state = self.state.copy()
        cv2.rectangle(current_state, (self.player_pos[1], self.player_pos[0]), (self.player_pos[3], self.player_pos[2]), (0,255,0), 2)
        cv2.rectangle(current_state, (self.target_pos[1], self.target_pos[0]), (self.target_pos[3], self.target_pos[2]), (0,0,255), 2)
        cv2.imwrite('state.png', current_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone', default='Android', choices=['Android', 'IOS'], type=str, help='OS')
    parser.add_argument('--resolution', default=[1280, 720], nargs=2, type=int, help='mobile phone resolution')
    parser.add_argument('--sensitivity', default=2.05, type=float)
    parser.add_argument('--resource', default='resource', type=str)
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    # print(args)

    AI = WechatAutoJump(args.phone, args.resolution, args.sensitivity, args.debug, args.resource)
    AI.run()
