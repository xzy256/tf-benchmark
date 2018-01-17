# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import gzip
import os
import tempfile

import time
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 导入数据
mnist = read_data_sets("mnist-data", one_hot=True)
# 定义参数
x = tf.placeholder(tf.float32, [None, 784])
Weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, Weights) + bias)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 定义交叉熵
# 使用梯度下降最小化loss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer() 
sess = tf.Session()
sess.run(init)  # 启动会话初始化数据
start = time.clock()
for i in range(1000):  # 开始训练
  # 获取train的image和label数据
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 执行train_step操作
  if i % 50 == 0:
     correct_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
     acc = tf.reduce_mean(tf.cast(correct_pre, "float"))
     print("step = %d, acc = %f" %(i, sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
end = time.clock()
print('time = ', end - start)
