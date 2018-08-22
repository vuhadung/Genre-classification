from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
#from tsne import bh_sne
import pickle
import sys
import os
from numpy import array
from processaudiofile import preprocessAudio

#The labels' name is assumed
labelsDict = {
    1:'blues',
    2:'classical',
    3:'country',
    4:'disco',
    5:'hiphop',
    6:'jazz',
    7:'metal',
    8:'pop',
    9:'reggae',
    10:'rock'
}

def getLabel(fname, audioname):
    for root, subdirs, files in os.walk(fname):
        for filename in files:
            if filename.endswith('train.csv'):
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        name,label = line.strip().split(',')
                        if name == audioname:
                            return label
    

def loadData(fname): 
    data = {}
    labels = []
    
    for root, subdirs, files in os.walk(fname):
        for filename in files:
            if filename.endswith('mp3'):
                file_path = os.path.join(root, filename)
                #print('\t- file %s (full path: %s)' % (filename, file_path))

                '''
                with open(file_path, 'r') as f:
                    try:
                        soundId = filename.split(".")[0]
                        content = f.read()
                        pp = pickle.loads(content)
                        pp = np.asarray(pp)
                        # pp = np.delete(pp, 1, axis=2)
                        data[soundId] = pp
                '''

                try:
                    soundId = filename.split(".")[0]
                    content = preprocessAudio(file_path)
                    data[soundId] = content

                    labelValue = getLabel(fname, filename)
                    #print (labelValue)
                    labelAsArray = [0] * len(labelsDict)
                    #print (labelAsArray)
                    labelAsArray[int(labelValue)-1] = 1
                    print (labelAsArray)
                    labels.append(labelAsArray)
                    #print (labels)
                except Exception as e:
                    print ("Error while loading data " + str(e))

    #save data and lables
    with open("data", 'wb') as f:
        f.write(pickle.dumps(list(data.values())))

    with open("labels", 'wb') as f:
        f.write(pickle.dumps(array(labels)))
    
    '''
    # print sizes
    print("Data set size: " + str(len(data.keys())))
    print("Number of genres: " + str(len(labelsDict.keys())))

    # convert image data to float64 matrix. float64 is need for bh_sne
    reshapedList = array(data.values())
    x_data = np.asarray(reshapedList).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE embedding
    vis_data = bh_sne(x_data, perplexity=30)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    colors = []
    for label in labels:
        colors.append(label.index(1))

    plt.scatter(vis_x, vis_y, c=colors, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10), label='Genres')
    plt.clim(-0.5, 9.5)
    plt.title('t-SNE MFCC samples as genres')
    plt.show()
    '''
