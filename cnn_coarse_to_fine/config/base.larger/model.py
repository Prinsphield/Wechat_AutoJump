import tensorflow as tf
import numpy as np
import os
import cv2

class JumpModel:
    def __init__(self):
        self.img_shape = (640, 720)
        self.batch_size = 8
        self.input_channle = 3
        self.out_channel = 2

    def conv2d(self, name, input, ks, stride):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name='%s-conv' % name)
                out = tf.nn.bias_add(out, b, name='%s-biad_add' % name)
        return out

    def make_conv_bn_relu(self, name, input, ks, stride, is_training):
        out = self.conv2d('%s-conv' % name, input, ks, stride)
        out = tf.layers.batch_normalization(out, name='%s-bn' % name, training=is_training)
        out = tf.nn.relu(out, name='%s-relu' % name)
        return out

    def make_fc(self, name, input, ks, keep_prob):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out = tf.matmul(input, w, name='%s-mat' % name)
                out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
                # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
        return out

    def forward(self, img, is_training, keep_prob):
        out = self.conv2d('conv1', img, [3, 3, self.input_channle, 16], 2)
        # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
        out = tf.nn.relu(out, name='relu1')

        out = self.make_conv_bn_relu('conv2', out, [3, 3, 16, 32], 1, is_training)
        out = self.make_conv_bn_relu('conv2_1', out, [3, 3, 32, 32], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        out = self.make_conv_bn_relu('conv3', out, [5, 5, 32, 64], 1, is_training)
        out = self.make_conv_bn_relu('conv3_1', out, [5, 5, 64, 64], 1, is_training)
        out = self.make_conv_bn_relu('conv3_2', out, [5, 5, 64, 64], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        out = self.make_conv_bn_relu('conv4', out, [7, 7, 64, 128], 1, is_training)
        out = self.make_conv_bn_relu('conv4_1', out, [7, 7, 128, 128], 1, is_training)
        out = self.make_conv_bn_relu('conv4_2', out, [7, 7, 128, 128], 1, is_training)
        out = self.make_conv_bn_relu('conv4_3', out, [7, 7, 128, 128], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        out = self.make_conv_bn_relu('conv5', out, [9, 9, 128, 256], 1, is_training)
        out = self.make_conv_bn_relu('conv5_1', out, [9, 9, 256, 256], 1, is_training)
        out = self.make_conv_bn_relu('conv5_2', out, [9, 9, 256, 256], 1, is_training)
        out = self.make_conv_bn_relu('conv5_3', out, [9, 9, 256, 256], 1, is_training)
        out = self.make_conv_bn_relu('conv5_4', out, [9, 9, 256, 256], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        out = tf.reshape(out, [-1, 256 * 20 * 23])
        out = self.make_fc('fc1', out, [256 * 20 * 23, 512], keep_prob)
        out = self.make_fc('fc2', out, [512, 128], keep_prob)
        out = tf.nn.dropout(out, keep_prob)
        out = self.make_fc('fc3', out, [128, 2], keep_prob)

        return out

if __name__ == '__main__':
    model = JumpModel()
    out = model.forward(tf.zeros((1, 640, 720, 3)))
    print(out.get_shape().as_list())

