# -*- coding: utf-8 -*-
"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/home/xzy/tf-input-data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 500  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10 
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
TRAIN_MAX = 5000


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False, "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


def data_type():
  """返回激活函数、权重、placeholder变量的类型"""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """如果数据不存在，则下载数据"""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """提取图像为4D tensor [image index, y, x, channels].
  其值从[0, 255]重新调整为[-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    """
    print('--------------------------------')
    print('data1=', data)
    print('--------------------------------')
    """
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    """
    print('################################')
    print('data2=', data)
    print('################################')
    """
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    """
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('data3=', data)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    """
    return data


def extract_labels(filename, num_images):
  """提取labels成一个int64 IDs的vector"""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """生成与MNIST的维度相匹配的假数据集"""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """基于密集预测和稀疏标签返回错误率"""
  return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) /
                  predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # 获取数据
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # 提取数据并转成numpy的array.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 1000)
    test_labels = extract_labels(test_labels_filename, 1000)

    # 产生一个验证集，并分片处理.VALIDATION_SIZE = 5000.
    validation_data = train_data[:VALIDATION_SIZE, ...]  # 0-5000分片
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:TRAIN_MAX+VALIDATION_SIZE, ...]
    train_labels = train_labels[VALIDATION_SIZE:TRAIN_MAX+VALIDATION_SIZE]
    num_epochs = NUM_EPOCHS   # NUM_EPOCHS = 10
  train_size = train_labels.shape[0]

  """ 训练样本和标签占位声明，每次训练都会喂一个批次的数据，
  通过向Run()方法传递{feed_dict}参数"""
  train_data_node = tf.placeholder(data_type(),
                                   shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(data_type(),
                             shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  """持久化训练权重，通过{tf.initialize_all_variables().run()}执行操作"""
  # 5x5 filter, depth 32.
  conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
                                                  stddev=0.1, seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))

  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  # fully connected, depth 512.
  fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1, seed=SEED, dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))

  fc2_weights = tf.Variable(tf.truncated_normal(
      [512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

  """训练子图和评估子图共用一个模型，通过共享训练参数"""
  def model(data, train=False):
    """2维卷积, 使用 'SAME'处理边界 (feature map输出和输入有相同的维数
       {strides} 是一个 4D array，他的形状为[image index, y, x, depth].
       data=[bitchSize=64, 28, 28, channels=1]
       conv1_weights=[5, 5, channels=1, filters=32]  
    
    print("---------initial data---------")
    print('data=', data)
    print('conv1_weights=', conv1_weights)
    """
    conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    """
    最大池化
    ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    因为我们不想在batch和channels上做池化
    """
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    """first conv-pool
       conv=[64, 28, 28, filters=32]       
       pool=[64, 14, 14, channels=32]
       conv2_weights=[5, 5, channels=32, filters=64]
    
    print('---------first conv-pool---------')
    print('pool=', pool)
    print('conv=', conv)
    print('conv2_weights=', conv2_weights)
    """
    conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """second conv-pool
       conv=[64, 14, 14, filters=64]
       pool=[64, 7, 7, channels=64]
    
    print('----------second conv-pool---------')
    print('conv=', conv)
    print('pool=', pool)
    """
    # 重构feature map成2D矩阵，传给全链接网络
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
                         [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    """
    print('-------reshape---------')
    print('pool_shape=', pool_shape)
    print('reshape=', reshape)
    print('fc1_weights=',fc1_weights)
    """
    # 全链接网络层
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # 训练数据时候，采用dropout随机丢弃50%的像素点
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  """ 训练计算公式: logits + cross-entropy loss.
      交叉熵：之前使用sigmoid{f(z) = 1/(1+\exp(-z))}函数计算神经元与真实值的欧式距离来判断偏差，
      反向去修正权重和偏置，但函数接近1的时候，导数会非常小，学习的速度就会变得非常缓慢。解决这个缺点
      就是使用交叉熵。参看http://blog.csdn.net/bixiwen_liu/article/details/52922008
  """
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=train_labels_node))

  # 防止过拟合，对全链接网络的参数进行正则化，常见的是dropout.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # 加入一个正则项.
  loss += 5e-4 * regularizers

  # 优化器: 每个批次都会增加，并且控制着学习率的衰变
  batch = tf.Variable(0, dtype=data_type())
  # 每次训练全部样本，学习率都会衰变，使用指数衰变
  learning_rate = tf.train.exponential_decay(
      0.01,                # 开始学习率
      batch * BATCH_SIZE,  # 数据集大小=批次×每批次数据大小.
      train_size,          # 训练多大更新学习率
      0.95,                # 衰变率--单位时间内衰变的几率.
      staircase=True)      # true代表训练train_size完更新一次，false代表每个样本都更新
  # 使用momentum（动量--它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向） 优化器.
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  #  计算当前训练minibatch的预测概率.
  train_prediction = tf.nn.softmax(logits)

  #  计算测试和验证minibatch上的预测概率
  eval_prediction = tf.nn.softmax(model(eval_data))

  # 通过feeding多个批次的数据到{eval_data},并从{eval_predictions}获取结果并保存.
  def eval_in_batches(data, sess):
    """获取一个小批次数据集的所有预测概率."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:  # 数据集小于每批次数据EVAL_BATCH_SIZE=64
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)  # NUM_LABELS=10
    for begin in xrange(0, size, EVAL_BATCH_SIZE):  # 每次增量为EVAL_BATCH_SIZE,一直到大于size
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(eval_prediction,
                                             feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(eval_prediction,
                                     feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # 创建session开始训练数据
  config = tf.ConfigProto(device_count={"CPU": 2},  # 4个cpu核
                          inter_op_parallelism_threads=1,
                          intra_op_parallelism_threads=4,
                          log_device_placement=True)
  st_time = time.clock()
  start_time = time.time()
#  with tf.Session() as sess:
  with tf.Session(config=config) as sess:
    # 初始化所有的训练参数
    tf.initialize_all_variables().run()
    print('Initialized!')
    # 根据 steps进行循环.
    # num_epochs = 10(非self_test)或 1(self_test)  BATCH_SIZE = 64 train_size=55000
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # 计算当前 minibatch 在数据集里的偏移.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # 向数据字典里喂数据
      feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
      # 取数据
      _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction],
                                       feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # 打印最终的结果
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)
  ed_time = time.clock()
  print('******total train time:', ed_time - st_time)


if __name__ == '__main__':
  tf.app.run()
