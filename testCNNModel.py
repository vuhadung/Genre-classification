from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from sklearn import svm
from processaudiofile import preprocessAudio

n_input = 599*13*5
testData = []

features_test = preprocessAudio("..\\Genre classification\\test\\685094653051980122.mp3") #label 2
features_test = np.asarray(features_test).reshape(n_input,)
testData.append(features_test)

features_test = preprocessAudio("..\\Genre classification\\test\\680312830197507790.mp3") #label 8
features_test = np.asarray(features_test).reshape(n_input,)
testData.append(features_test)

features_test = preprocessAudio("..\\Genre classification\\test\\683065700950296145.mp3") #label 5
features_test = np.asarray(features_test).reshape(n_input,)
testData.append(features_test)

features_test = preprocessAudio("..\\Genre classification\\test\\685052294264222117.mp3") #label 4
features_test = np.asarray(features_test).reshape(n_input,)
testData.append(features_test)

testData = np.asarray(testData)
testData = testData.reshape((testData.shape[0], n_input))
print(testData.shape)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph("..\Genre classification\model.final.meta")
    # Restore variables from disk.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print("Model restored.")
	
    graph = tf.get_default_graph()
    x= graph.get_tensor_by_name("x:0")
    keep_prob= graph.get_tensor_by_name("keep_prob:0")
    pred = graph.get_tensor_by_name("pred:0")

    for op in graph.get_operations():
        if op.type == "Placeholder":
            print(op.name)
    
    out = sess.run(pred, feed_dict={x: testData, keep_prob: 1.})
    out = tf.argmax(out,1).eval()
    print(out)
