from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# f.logging.set_verbosity(tf.logging.INFO)
# model settings
enhancer_length = 3000
promoter_length = 2000
n_kernels = 200
filter_length = 40
dense_layer_size = 800
#import matplotlib as mp
#mp.use('TkAgg')
#import matplotlib.pyplot as plt

# Convolutional/ maxpooling layers to extract prominet motif
import argparse
import numpy as np
import h5py
#import cv2
import sys
#import tempfile

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
def balance_dataset(x,labels):
  # padding the class 1 data by replicating themselves to make a balance dataset
  idx_c1 = sum(labels)
  idx_total = labels.shape[0]
  idx_padd = np.zeros((idx_c1))
  t = 0
  for i in range(idx_total):
    if labels[i]==1:
      idx_padd[t] = i
      t = t+1
  idx_need = idx_total-idx_c1
  r = int(float(idx_need)/float(idx_c1))
  res = idx_need-r*idx_c1
  paddings = x[idx_padd, :, :]
  paddingsL = np.ones((idx_need))
  for j in range(r):
    x = np.concatenate((x,paddings),axis=0)

  x = np.concatenate((x,paddings[0:res,:,:]),axis=0)
  labels = np.concatenate((labels,paddingsL),axis=0)
  return x, labels

def get_batch_data(x, x_infer, labels,batchsize):
    data_length = x.shape[0]
    idx_start = np.random.randint(0,data_length-batchsize,size=1)
    batch_x = x[idx_start:idx_start+batchsize,:,:]
    batch_x_infer = x_infer[idx_start:idx_start+batchsize,:,:]
    batch_labels = labels[idx_start:idx_start+batchsize]

    return batch_x, batch_x_infer, batch_labels

def deepnn(x,x_infer):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('sequence'):
    x_image1 = tf.reshape(x, [-1, 3000, 4])
    print("Input format: ")
    print(x)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([40, 4, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv1d(x_image1, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_conv1 = tf. reshape(h_conv1, [-1, 3000, 8, 1])
    h_pool1 = max_pool_1D(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    h_pool1 = tf.reshape(h_pool1,[-1,3000,8])
    W_conv2 = weight_variable([40, 8, 4])
    b_conv2 = bias_variable([4])
    h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_conv2 = tf.reshape(h_conv2, [-1,3000, 4, 1])
    h_pool2 = max_pool_1D(h_conv2)

  # The other network for the sequence 2
  with tf.name_scope('sequence2'):
    x_image2 = tf.reshape(x_infer, [-1, 2000, 4])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv21'):
    W_conv21 = weight_variable([40,4,8])
    b_conv21 = bias_variable([8])
    h_conv21 = tf.nn.relu(conv1d(x_image2, W_conv21) + b_conv21)


  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool21'):
    h_conv21 = tf.reshape(h_conv21, [-1, 2000, 8, 1])
    h_pool21 = max_pool_1D(h_conv21)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv22'):
    h_pool21 = tf.reshape(h_pool21, [-1,2000,8])
    W_conv22 = weight_variable([40,8,4])
    b_conv22 = bias_variable([4])
    h_conv22 = tf.nn.relu(conv1d(h_pool21, W_conv22) + b_conv22)

  # Second pooling layer.
  with tf.name_scope('pool22'):
    h_conv22 = tf.reshape(h_conv22,[-1, 2000, 4,1])
    h_pool22 = max_pool_1D(h_conv22)
# combine two convolution networks(enhancer/promoter) to the fully connect network
  h_poolC = tf.concat((h_pool2,h_pool22),axis=1)
  print("combine shape: ")
  print(h_poolC.__sizeof__())
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([4*(3000+2000), 5000])
    b_fc1 = bias_variable([5000])

    h_pool2_flat = tf.reshape(h_poolC, [-1, 4*(3000+2000)])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([5000, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([1024,256])
    b_fc3 = bias_variable([256])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
  with tf.name_scope('fc4'):
    W_fc4 = weight_variable([256,1])
    b_fc4 = bias_variable([1])

    y_conv = tf.sigmoid(tf.matmul(h_fc3, W_fc4) + b_fc4)
    print('Output of the last layer')
    print(y_conv)
  return y_conv, keep_prob

def conv1d(x, W):
   """conv1d returns a 1d convolution layer with full stride."""
   return tf.nn.conv1d(x, W, stride = 1, padding='SAME')


def max_pool_1D(x):
  """max_pool_2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                        strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir)
  #cell_lines = ['GM12878', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
  cell_lines = ['GM12878']
  #data_path = '/home/panwei/zhuan143/all_cell_lines/'
  data_path = '/Users/tczz/Desktop/'
  for cell_line in cell_lines:
      print('Loading ' + cell_line + ' data from ' + data_path)
      X_enhancers = None
      X_promoters = None
      labels = None
      file_path = (data_path+cell_line)
      '''
      with h5py.File(data_path, 'r') as hf:
          X_enhancers = np.array(hf.get( 'X_enhancers'))
          X_promoters = np.array(hf.get('_X_promoters'))
          labels = np.array(hf.get('Labels'))'''
      X_enhancers = np.load(data_path+cell_line+'_enhancers_small.npy')
      X_promoters = np.load(data_path+cell_line+'_promoters_small.npy')
      labels = np.load(data_path+cell_line+'_labels_small.npy')

  #X_enhancers = tf.convert_to_tensor(X_enhancers, dtype=tf.float32)
  #X_promoters = tf.convert_to_tensor(X_promoters, dtype=tf.float32)
  #labels = tf.convert_to_tensor(labels,dtype=tf.int32)
  # combine all the cell lines or one for each
  # test on the simplest case, with one cell line
  # divide the data into train and test
  training_idx = np.random.randint(0,int(0.8*X_enhancers.shape[0]), size=4)#size=int(0.64*X_enhancers.shape[0]))
  test_idx = np.random.randint(int(0.8*X_enhancers.shape[0])+1,X_enhancers.shape[0], size=2)#size=int(0.16*X_enhancers.shape[0]))

  Enhancers_train, Enhancers_test = X_enhancers[training_idx, :, :], X_enhancers[test_idx, :, :]
  Promoters_train, Promoters_test = X_promoters[training_idx, :, :], X_promoters[test_idx, :, :]
  Labels_train, Labels_test = labels[training_idx], labels[test_idx]
  labels_back = Labels_train
  Enhancers_train, Labels_train = balance_dataset(Enhancers_train, Labels_train)
  Promoters_train, Labels_train = balance_dataset(Promoters_train, labels_back)

  labels_back2  = Labels_test
  Enhancers_test, Labels_test = balance_dataset(Enhancers_test, Labels_test)
  Promoters_test, Labels_test = balance_dataset(Promoters_test, labels_back2)

  # Create the model
  x = tf.placeholder(tf.float32, [None, enhancer_length,4], name="x-input")
  x_infer = tf.placeholder(tf.float32, [None, promoter_length,4], name="x-infer-input")
  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None], name="y-input")
  #y_infer = tf.placeholder(tf.int64, [None])
  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x,x_infer)
  print('The logits:')
  print(y_conv)
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  #tf.scalar_summary("loss", cross_entropy)
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    print('Prediction:')
    print(y_conv)
    print(y_)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    print(tf.argmax(y_conv,1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      #enhancer_batch, promoter_batch, labels_batch = tf.train.batch([Enhancers_train,Promoters_train,Labels_train],2)
      #promoter_batch = tf.train.batch(Promoters_train,50)
      #labels_batch = tf.train.batch(labels,50)
      '''sess = tf.Session()
      print(enhancer_batch)
      print(promoter_batch)
      print(labels_batch)
      #with tf.Session():'''


      enhancer_batch_d, promoter_batch_d, labels_batch_d = get_batch_data(Enhancers_train, Promoters_train, Labels_train)
      labels_batch_d_back = labels_batch_d
      enhancer_batch_d, labels_batch_d = balance_dataset(enhancer_batch_d, labels_batch_d)
      promoter_batch_d, labels_batch_d = balance_dataset(promoter_batch_d, labels_batch_d_back)
      if i % 1 == 0:
        print(i)
        train_accuracy = accuracy.eval(feed_dict={x: enhancer_batch_d, x_infer: promoter_batch_d, y_: labels_batch_d, keep_prob: 1.0})
        #print('max: %g, min: %g' % (np.max(batch[0]), np.min(batch[0])))
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: enhancer_batch_d, x_infer: promoter_batch_d, y_: labels_batch_d, keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: Enhancers_test, x_infer: Promoters_test, y_: Labels_test, keep_prob: 1.0}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  tf.app.run()
