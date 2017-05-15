# -*- coding: utf-8 -*-
"""
name:           soundtypes_deep.py
author:         mattia paterna
created:        march, 21 2017
last edit:
purpose:        barebone implementation of sound-types using deep learning
"""


# import modules and functions
import tensorflow as tf
from utils import compute_spectrogram, convnet


# general parameters
INPUT_FILE = 'samples/Jarrett_Vienna_cut.wav'           # input file for feature learning
NO_MELS = 128                                           # no. of mel bins
FFT_SIZE = 1024                                         # size of FFT
HOP_SIZE = 512                                          # hop size (overlap)
DURATION = 10                                           # segment duration (sec)

if __name__ == "__main__":
    print ('[soundtypes - feature learning using convnets]\n')
    print('- computing features from the input signal -')

    # - feature extraction stage -

    # time-frequency representation
    [S, DIMS] = compute_spectrogram(
        audiofile=INPUT_FILE,
        no_mels=NO_MELS,
        fft_size=FFT_SIZE,
        hop_size=HOP_SIZE,
        duration=DURATION
    )


    # - feature learning stage -

    print('- feature learning using convolutional neural network - ')
    # nodes for the input and output of the network graph
    # obs: the use of [None, None] requires later specifications
    X = tf.placeholder(tf.float32, shape=[None,None],name='spectrogram')
    Y_ = tf.placeholder(tf.float32, shape=[None, None],name='logits')


    # train the convnet
    Y_ = convnet(
        x=X,
        dims=DIMS
    )



# to do
# 1. create minibatch for spectrograms (?)
# 2. allow for training every 10 secs or so when a new spectogram is computed
# 3. batch normalization, removing biases and dropout?