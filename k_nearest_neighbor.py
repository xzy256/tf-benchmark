# -*- coding: utf-8 -*-
# k领近分类模型
from __future__ import division
import gzip
import tempfile
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import os
import tensorflow as tf
import time
os.environ['TF_CPP_LOG_LEVEL']='2'

#load dataset,one_ho编码代表的是只有一个位代表1，其他位为0
mnist = read_data_sets("mnist-data", one_hot = True)

#我们对mnist数据集做一个数据限制
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(1000)

#计算图输入占位符
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [784])

#使用L1距离进行最近计算，axis = 1 代表行与行相加，行数不降维
distance = tf.reduce_sum(tf.abs(tf.add(x, tf.negative(y))), axis=1)
#预测
pred = tf.arg_min(distance, 0)
#最近部分类的准确率
accuracy = 0
#初始化节点
init = tf.global_variables_initializer()

#启动会话
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xtest)
    start = time.clock()
    for i in range(Ntest):
        #feed_dict 数据字典，train全部数据，每次test 1个
        nn_index = sess.run(pred, feed_dict={x: Xtrain, y:Xtest[i, :]})
        pred_class_label = np.argmax(Ytrain[nn_index])
        true_class_label = np.argmax(Ytest[i])
        #计算准确率
        if pred_class_label == true_class_label:
            accuracy += 1
        if i % 50 == 0:
            acc = accuracy / Ntest
            print("step=%d, acc=%f"%(i, acc))
    end = time.clock()
    print('训练时间为: %s 秒'%(end-start))
    print("Done!")
