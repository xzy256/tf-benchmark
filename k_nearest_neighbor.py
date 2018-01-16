# -*- coding: utf-8 -*-
# k领近分类模型
import numpy as np
import os
import tensorflow as tf
import time
os.environ['TF_CPP_LOG_LEVEL']='2'

#load dataset,one_ho编码代表的是只有一个位代表1，其他位为0
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/xzy/tf-input-data", one_hot = True)

#我们对mnist数据集做一个数据限制
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)
print('Xtrain.shape: ', Xtrain.shape, ', Ytest.shape: ',Xtest.shape)
print('Ytrain.shape: ', Ytrain.shape, ', Ytest.shape: ',Ytest.shape)

#计算图输入占位符
xtrain = tf.placeholder("float", [None, 784])
xtest = tf.placeholder("float", [784])

#使用L1距离进行最近计算，axis = 1 代表行与行相加，行数不降维
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)
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
        nn_index = sess.run(pred, feed_dict={xtrain: Xtrain, xtest:Xtest[i, :]})
        pred_class_label = np.argmax(Ytrain[nn_index])
        true_class_label = np.argmax(Ytest[i])
        print("Test", i, "Predicted Class Label:", pred_class_label,"True Class Label:", true_class_label)
        #计算准确率
        if pred_class_label == true_class_label:
            accuracy += 1
    end = time.clock()
    print('训练时间为: %s 秒'%(end-start))
    print("Done!")
    print('--------------->',accuracy)
    print('--------------->',Ntest)
    accuracy /= Ntest
    print("Accuracy:%f"%accuracy)
