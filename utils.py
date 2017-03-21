# -*- coding: utf-8 -*-
"""
name:           utils.py
author:         mattia paterna
created:        march, 16 2017
last edit:      march, 21 2017
purpose:        contains utilities for feature learning using deep architectures
"""

# import needed modules
import numpy as np                      # general computation
import librosa                          # audio signal processing, feature extraction
from librosa import display             # plot the spectrogram
import matplotlib.pyplot as plt         # general plotting
import tensorflow as tf                 # neural networks

# - Feature Extraction -
# this function computes the spectrogram over 10 secs of a file
# and return the spectrogram S as matrix and its dimensions
def compute_spectrogram(audiofile):
    [y, sr] = librosa.core.load(audiofile, sr=None, mono=True)

    # get audio file first 10 secs
    len = sr * 10
    y = y[0:len]

    # get options for computing spectrogram (maybe as input variable?)
    mels = 128                          # frequency resolution
    fft = 1024                          # fft length
    hop = 512                           # hop size

    # Mel spectrogram for the file, overlap=50%
    S = librosa.feature.melspectrogram(y, sr, n_fft=fft, n_mels=mels, hop_length=hop)
    # log-compressed version
    S = librosa.power_to_db(S, ref=np.max)

    # maybe adding dynamic range compression stage? (Dieleman, 2014)
    # (C = amount of compression, set by heuristics)
    # C = 10000
    # S = np.log(1 + C*S)

    # get spectrogram matrix dimensions (to be used in the convnet)
    dims = S.shape

    # plot the spectrogram
    plt.figure(figsize=[12, 4])
    librosa.display.specshow(S, sr=sr, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

    return S, dims


# - Feature learning using deep architectures -
# this function initialises the weights for the convnet
def weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


# this function initialises the biases
# (slightly positive to avoid dead neurons at the start)
def bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# this function build the graph for the convolutional neural network
# that is used for feature learning from the input spectrogram
def convnet()

