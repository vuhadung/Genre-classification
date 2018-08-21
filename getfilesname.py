import os
import csv
import librosa

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

if __name__ == "__main__":
    audioList = []
    for root, dirs, files in os.walk("..\Genre Classification"):
        #print ("Root is " + root)
        for name in files:
            #print(os.path.join(root, name))
            #tmp = name.split(".")[0]
            #print (tmp)
            if name.endswith('mp3'):
                #audioList.append(name)

                #Test librosa lib
                fpath = os.path.join(root, name)
                print(fpath)
                y, src = librosa.load(fpath, duration=30)    
                
            #if name.endswith('train.csv'):
                #print (name)

    #label = getLabel("..\Genre Classification", "104185445737176022.mp3")
    #print (label)
    
    '''
    with open('data.csv', 'a') as file:
        for l in audioList:
            file.write(l)
            file.write('\n')
    '''
