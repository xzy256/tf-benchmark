# -*- coding: utf-8 -*-
#一元线性回归
import os
import time
import tensorflow as tf
import numpy as np
#from matplotlib import pylab as plt
os.environ['TF_CPP_LOG_LEVEL']='2'

#产生数据评估集
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.166,9.779,6.182,7.89,2.167,7.042,10.791,
                      5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                      2.827,3.465,1.65,2.904,2.42,2.92,1.3])
n_train_examples = train_X.shape[0]
#产生测试样本
test_X = np.asarray([6.83,4.668,5.9,7.91,5.7,8.7,3.1,2.1])
test_Y = np.asarray([1.84,2.273,3.2,2.831,2.92,3.24,1.35,1.03])
n_test_examples = test_X.shape[0]

#展示原始数据分析，调用的是matlab的函数
#plt.plot(train_X, train_Y, 'ro', label='Original train points')
#plt.plot(test_X, test_Y, 'b*', label='original test points')
##添加图例的标注
#plt.legend()
#plt.show()

with tf.Graph().as_default():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, name='X')
        Y_true = tf.placeholder(tf.float32, name='Y_true')
    with tf.name_scope('inference'):
        #模型参数变量
        w = tf.Variable(tf.zeros([1]), name='weight')
        b = tf.Variable(tf.zeros([1]), name='bias')
        #inference y = wx + b
        Y_pred = tf.add(tf.multiply(X, w), b)
    with tf.name_scope('Loss'):
        #添加损失,最小二乘法选取拟合曲线
        TrainLoss = tf.reduce_mean(tf.pow(Y_true - Y_pred, 2))/2
    with tf.name_scope('Train'):
        #创建优化器
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
        TrainOp = Optimizer.minimize(TrainLoss)
    with tf.name_scope('Evaluate'):
        # 添加评估
        EvalLoss = tf.reduce_mean(tf.pow(Y_true - Y_pred, 2)) / 2
    #添加出事化节点
    initOp = tf.global_variables_initializer()
    #保存计算图
    writer = tf.summary.FileWriter(logdir='logs',graph=tf.get_default_graph())
    writer.close()
    #启动会话
    print('开始启动会话，执行计算图，开始训练')
    start = time.clock()
    sess = tf.Session()
    sess.run(initOp)

    for step in range(1000):
        _, train_loss, train_w, train_b = sess.run([TrainOp, TrainLoss, w, b],feed_dict={X: train_X, Y_true: train_Y})

#        if step % 50 == 0:
#            print('step=%d train_loss=%f' % (step, train_loss))
        if  step % 50 == 0:
            test_loss = sess.run(EvalLoss, feed_dict={X: test_X, Y_true: test_Y})
	    print('step=%d acc=%f' % (step, 1 - test_loss))

    end = time.clock()
    print('total train time=',end-start)
    print("end")
#    w, b = sess.run([w, b])
#   print("end train", "w=", w, "b=", b)
    # 展示拟合曲线
#    plt.plot(train_X, train_Y, 'ro', label='Original train points')
#    plt.plot(test_X, test_Y, 'b*', label='original test points')
#    plt.plot(train_X, w*train_X + b, label='Fitted line')
#    plt.legend()
#    plt.show()
