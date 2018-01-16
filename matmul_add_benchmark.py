# --* coding: UTF-8 *--
# 做一个矩阵乘
import tensorflow as tf
import numpy as np
import time


step = 1000
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
ouput = tf.matmul(input1, input2)

with tf.Session() as sess:
    for i in range(1, 11):
        index = i * step
        x = np.float32(np.random.rand(index, index))  # shape=(20, index)
        y = np.float32(np.random.rand(index, index))

        start_time = time.time()
        sess.run(ouput, feed_dict={input1: x, input2: y})
        duration = time.time() - start_time
        print ('step %d, duration = %.3f') %(index, duration)
