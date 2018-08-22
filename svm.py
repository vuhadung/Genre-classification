from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from sklearn import svm
from processaudiofile import preprocessAudio

n_input = 599 * 13 * 5
n_classes = 10
    
# Load data
data = []
with open("data", 'rb') as f:
    content = f.read()
    data = pickle.loads(content)
data = np.asarray(data)
data = data
data = data.reshape((data.shape[0], n_input))

labels = []
with open("labels", 'rb') as f:
    content = f.read()
    labels = pickle.loads(content)

new_labels = []
for i in range(0, 100):
    tmp_data = labels[i]
    tmp_label = [i for i, j in enumerate(tmp_data) if j == 1]
    new_labels.append(tmp_label[0] + 1)
new_labels = np.asarray(new_labels)

model = svm.SVC()
model.fit(data, new_labels)

features_test = preprocessAudio("..\\Genre classification\\test\\685094653051980122.mp3") #label 2
features_test = np.asarray(features_test).reshape(n_input,)
features_test = features_test.reshape(1, -1)
predicted_labels = model.predict(features_test)
print (predicted_labels)

features_test = preprocessAudio("..\\Genre classification\\test\\680312830197507790.mp3") #label 8
features_test = np.asarray(features_test).reshape(n_input,)
features_test = features_test.reshape(1, -1)
predicted_labels = model.predict(features_test)
print (predicted_labels)

features_test = preprocessAudio("..\\Genre classification\\test\\683065700950296145.mp3") #label 5
features_test = np.asarray(features_test).reshape(n_input,)
features_test = features_test.reshape(1, -1)
predicted_labels = model.predict(features_test)
print (predicted_labels)

features_test = preprocessAudio("..\\Genre classification\\test\\685052294264222117.mp3") #label 4
features_test = np.asarray(features_test).reshape(n_input,)
features_test = features_test.reshape(1, -1)
predicted_labels = model.predict(features_test)
print (predicted_labels)

