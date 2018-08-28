from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# f.logging.set_verbosity(tf.logging.INFO)
# model settings
'''enhancer_length = 3000
promoter_length = 2000
n_kernels = 200
filter_length = 40
dense_layer_size = 800'''
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt

# Convolutional/ maxpooling layers to extract prominet motif
import argparse
import numpy as np
#import cv2
import sys
#import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    '''print("List length:")
    print(len(X_imgs))'''
    #print(X_imgs.eval())

    X_imgs_copy = np.reshape(X_imgs.copy(),(-1,28,28))

    '''print(X_imgs_copy[0].shape)'''
    row, col = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        #print(coords[0])
        #print(coords[1])
        X_img[coords[0], coords[1]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1]] = 0
    X_imgs_copy = np.reshape(X_imgs_copy,(-1,784))
    return X_imgs_copy

'''
def add_gaussian_noise(X_imgs):
    X_imgs_copy = np.reshape(X_imgs.copy(),(-1,28,28))
    gaussian_noise_imgs = []
    row, col= X_imgs_copy[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs_copy:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        #gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.15, 0.25 * gaussian, 0.85, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    gaussian_noise_imgs = np.reshape(gaussian_noise_imgs,(-1,784))
    return gaussian_noise_imgs'''


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
    x_image1 = tf.reshape(x, [-1, 28, 28, 1])
    print("Input format: ")
    print(x)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 10])
    b_conv1 = bias_variable([10])
    h_conv1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 10, 20])
    b_conv2 = bias_variable([20])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # The other network for the sequence 2
  with tf.name_scope('sequence2'):
    x_image2 = tf.reshape(x_infer, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv21'):
    W_conv21 = weight_variable([5, 5, 1, 10])
    b_conv21 = bias_variable([10])
    h_conv21 = tf.nn.relu(conv2d(x_image2, W_conv21) + b_conv21)


  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool21'):
    h_pool21 = max_pool_2x2(h_conv21)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv22'):
    W_conv22 = weight_variable([5, 5, 10, 20])
    b_conv22 = bias_variable([20])
    h_conv22 = tf.nn.relu(conv2d(h_pool21, W_conv22) + b_conv22)

  # Second pooling layer.
  with tf.name_scope('pool22'):
    h_pool22 = max_pool_2x2(h_conv22)
# combine two convolution networks(enhancer/promoter) to the fully connect network
  h_poolC = tf.concat((h_pool2,h_pool22),axis=1)
  print("combine shape: ")
  #print(h_poolC.__sizeof__())
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([8 * 4 * 20, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_poolC, [-1, 8* 4 * 20])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

def conv2d(x, W):
   """conv2d returns a 2d convolution layer with full stride."""
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


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
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="x-input")
  x_infer = tf.placeholder(tf.float32, [None, 784], name="x-infer-input")
  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None], name="y-input")
  #y_infer = tf.placeholder(tf.int64, [None])
  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x,x_infer)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  #tf.scalar_summary("loss", cross_entropy)
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  #tf.scalar_summary("accuracy", accuracy)
  #summary_op = tf.merge_all_summaries()

  #graph_location = tempfile.mkdtemp()
  #print('Saving graph to: %s' % graph_location)
  #train_writer = tf.summary.FileWriter(graph_location)
  #train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      x_infer_batch = add_salt_pepper_noise(batch[0])
      #x_infer_batch = add_gaussian_noise(batch[0])
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], x_infer: x_infer_batch, y_: batch[1], keep_prob: 1.0})
        #print('max: %g, min: %g' % (np.max(batch[0]), np.min(batch[0])))
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], x_infer: x_infer_batch, y_: batch[1], keep_prob: 0.5})
    x_infer_test = add_salt_pepper_noise(mnist.test.images)
    #x_infer_test = add_gaussian_noise(mnist.test.images)
    #plt.figure()
    #plt.imshow(np.reshape(x_infer_test[0],(28,28)))
    #plt.show()
    #plt.figure()
    #plt.imshow(np.reshape(mnist.test.images[0],(28,28)))
    #plt.show()
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: x_infer_test, x_infer: x_infer_test, y_: mnist.test.labels, keep_prob: 1.0}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
    #file = open("training_process.txt")
    #file.write("test accuracy:"+str(accuracy.eval(feed_dict={
        #x: x_infer_test, x_infer: x_infer_test, y_: mnist.test.labels, keep_prob: 1.0})))
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  tf.app.run()
