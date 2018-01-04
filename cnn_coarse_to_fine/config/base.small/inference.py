import sys
sys.path.append('/home/anjie/projects/we_jump/')
import numpy as np
import os
import tensorflow as tf
from data_provider.jump_data import JumpData
from model import JumpModel
from tqdm import tqdm
import argparse, cv2
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ""

def get_a_img(path):
    # name is img path
    img = cv2.imread(path)
    return img[np.newaxis, :, :, :]

def inference(path):
    net = JumpModel()
    img = tf.placeholder(tf.float32, [None, 640, 720, 3], name='img')
    label = tf.placeholder(tf.float32, [None, 2], name='label')
    is_training = tf.placeholder(np.bool, name='is_training')
    keep_prob = tf.placeholder(np.float32, name='keep_prob')
    lr = tf.placeholder(np.float32, name='lr')

    pred = net.forward(img, is_training, keep_prob)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('./train_logs')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('==== successfully restored ====')

    val_img = get_a_img(path)
    feed_dict = {
        img: val_img,
        is_training: False,
        keep_prob: 1.0,
    }
    pred_out = sess.run(pred, feed_dict=feed_dict)
    return pred_out

if __name__ == '__main__':
    dataset = JumpData()
    name = dataset.val_name_list[0]
    posi = name.index('_res')
    img_name = name[:posi] + '.png'
    a = time.time()
    pred = inference(img_name)
    print(pred, time.time() - a)
