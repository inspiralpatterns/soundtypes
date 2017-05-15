"""
name:           create_sine_dataset.py
author:         mattia paterna
created:        May, 5th 2017
last edit:
purpose:        create a custom dataset based on sine_dataset by-purpose.
                This dataset is a collection of spectrograms made over 1-sec samples
                using specific configuration parameters given in the method
                `compute_spectorgram`.
                The dataset is then saved inside a multi-dimensional numpy array.
"""

# import modules
import numpy as np
import librosa
from datetime import datetime

# define function for computing spectrograms
def compute_spectrogram(audiofile, sr, no_mels, fft_size, hop_size):

    y = audiofile
    sr = sr

    # get options for computing spectrogram
    mels = no_mels                      # frequency resolution
    fft = fft_size                      # fft length
    hop = hop_size                      # hop size

    # Mel spectrogram for the file, overlap=50%
    S = librosa.feature.melspectrogram(y, sr, n_fft=fft, n_mels=mels, hop_length=hop)
    # log-compressed version
    S = librosa.power_to_db(S, ref=np.max)

    # get spectrogram matrix dimensions (to be used in the convnet)
    dims = S.shape

    return S, dims


# general parameter initialisation
sr = 44100                                              # sampling rate
fft_size = 1024                                         # size of FFT
hop_size = 512                                          # hop size (overlap)
duration = 1                                            # segment duration (sec)
no_mels = 128                                           # no. of mel bins
no_fft = 86                                             # frames in spectrogram
chunk_len = sr * duration

# load the dataset structure
path = 'sine_dataset.npz'
data = np.load(path, 'rb')
no_spectrograms = len(data.files)                       # total number of spectrograms
ids = data.files                                        # get the id for each recording (list)
print(len(ids), ' sinusoids in the dataset')

# variables initialisation
dataset = np.empty([no_mels, no_fft, 0])                # empty 3-d array

X = np.empty(chunk_len)                                 # store 1-sec sine samples


# SPECTROGRAM COMPUTATION
print('Computation started at ', datetime.now().time())
for id in ids:

    # extract the corresponding audio recording
    X = data[id][0]

    # compute the spectrogram for the sample
    [S, dims] = compute_spectrogram(
        audiofile=input,
        sr=sr,
        no_mels=no_mels,
        fft_size=fft_size,
        hop_size=hop_size,
    )

    # reshape spectrogram
    S = np.reshape(S, newshape=[dims[0], dims[1], 1])
    dataset = np.concatenate((dataset, S), axis=2)

# DATASET STORAGE
# save the dataset
np.savez_compressed(file='sine_spec_dataset', dataset=dataset)
print('Saved and successfully finished at ', datetime.now().time())

