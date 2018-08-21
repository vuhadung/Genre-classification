import numpy as np
import pickle
import sys
import os
import librosa

SOUND_SAMPLE_LENGTH = 30000

HAMMING_SIZE = 100
HAMMING_STRIDE = 40

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def preprocessAudio(audioPath):
    print ('Pre-process ' + audioPath)

    '''
    #tricky part - need improvement
    audioPath = audioPath.replace('..','')
    absolutePath = "D:\PROJECTS" + audioPath
    print (absolutePath)
    '''

    featuresArray = []
    try: 
        for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
            if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:

                y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)

                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

                # Convert to log scale (dB). We'll use the peak power as reference
                log_S = librosa.power_to_db(S)

                mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
                featuresArray.append(mfcc)

                # featuresArray.append(S)
                if len(featuresArray) == 599:
                    break

    except Exception as e:
        print ("Error while preprocessing audio " + str(e))

    return featuresArray

    '''            
    ppFilePath = rreplace(audioPath, ".au", ".pp", 1)
    print 'storing pp file: ' + ppFilePath

    f = open(ppFilePath, 'w')
    f.write(pickle.dumps(featuresArray))
    f.close()
    '''
