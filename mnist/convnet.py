from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # labels are on_hot encoded

x = tf.placeholder(tf.float32, shape=[None, 784])  # since image is 28 * 28
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # ten classes for the arabic numerals


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# We move a 5 * 5 pixel over the image, which has a padding of two
# on the sides, and top and bottom of the image, Meaning the center
# square in the 5 * 5 grid starts on the top-left pixel of the original
# image.
#
# After the first convolusion layer, we have transformed from a
# 28 * 28 * 1 tensor -> 28 * 28 * 32 tensor
#
# A max pooling layer is then applied where the max value in a
# 2 * 2 * 1 grid is taken, resulting in a 14 * 14 * 32 tensor

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Another convolusion layer is applied, producing a 14 * 14 * 64 tensor
# with the same 5 * 5 grid and padding. The same max pooling layer is then
# applied, resulting in a 7 * 7 * 64 tensor.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# We have a fully connected layer that produces a vector with length 1024.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Here is a dropout layer that is only used in training. Some output neurons
# will "randomly" not fire. This can be thought of as helping the network
# learn all aspects of the model and not just learn the obvious one.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# This is the final step, producing a vector of length ten. Where each element
# represents one of our ten classes (numbers)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Probabilities are computed using the softamx function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Use the Adam Optimiser (a gradient based optimiser)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# finds the element with the max value for each vector and whether they are equal
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# computes the mean across all numbers (1 is correct, 0 is incorrect)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
