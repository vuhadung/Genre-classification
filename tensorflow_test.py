from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from sklearn import svm
from processaudiofile import preprocessAudio

def conv2d(sound, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(sound, k):
    return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

sess = tf.InteractiveSession()
	
n_input = 599*13*5
n_classes = 10
_dropout = 0.75
features_test = preprocessAudio("..\\Genre classification\\test\\685094653051980122.mp3") #label 2
features_test = np.asarray(features_test).reshape(n_input,)

_weights = {
        # 4x4 conv, 1 input, 149 outputs
        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        # 4x4 conv, 149 inputs, 73 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        # 4x4 conv, 73 inputs, 35 outputs
        'wc3': tf.Variable(tf.random_normal([2, 2, 73, 35])),
        # fully connected, 38*8*35 inputs, 2^13 outputs
        'wd1': tf.Variable(tf.random_normal([75 * 2 * 73, 1024])),
        # 2^13 inputs, 13 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

_biases = {
        'bc1': tf.Variable(tf.random_normal([149])),
        'bc2': tf.Variable(tf.random_normal([73])),
        'bc3': tf.Variable(tf.random_normal([35])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

# Construct model
x = tf.reshape(features_test, shape=[-1, 599, 13, 5])
x=tf.cast(x, tf.float32)

# Convolution Layer
conv1 = conv2d(x, _weights['wc1'], _biases['bc1'])
# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=4)
# Apply Dropout
conv1 = tf.nn.dropout(conv1, _dropout)

# Convolution Layer
conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)
# Apply Dropout
conv2 = tf.nn.dropout(conv2, _dropout)
# Fully connected layer
# Reshape conv3 output to fit dense layer input
dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
# Relu activation
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
# Apply Dropout
dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

# Output, class prediction
out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
label = tf.argmax(out, 1) # give index of the largest value 

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(out))

predicted_label = sess.run(label)
print(predicted_label[0]+1)
sess.close()
