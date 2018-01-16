# coding=utf-8

import tensorflow as tf
import time
# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/xzy/tf-input-data", one_hot=True)
# 定义参数
x = tf.placeholder(tf.float32, [None, 784])
Weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, Weights) + bias)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 定义交叉熵
# 使用梯度下降最小化loss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()  # 定义初始化操作
sess = tf.Session()
sess.run(init)  # 启动会话初始化数据
start = time.clock()
for i in range(1000):  # 开始训练
  # 获取train的image和label数据
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 执行train_step操作
end = time.clock()
print('time = ', end - start)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 定义准确度预测操作
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
