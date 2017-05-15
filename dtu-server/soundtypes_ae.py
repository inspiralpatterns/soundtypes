# -*- coding: utf-8 -*-
"""
name:           soundtypes_ae.py
author:         mattia paterna
created:        April, 8 2017
last edit:
purpose:        implementation of a deep unsupervised structure using different types of autoencoder
"""

import numpy as np
from keras.layers import Input # define the input shape for the model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D # for the convnet structure
from keras.models import Model # for the overall definition

# DEEP LEARNING PART

# load the training dataset
dataset = np.load('dataset_test.npz','r')
data_training = dataset['dataset']
data_training = np.transpose(data_training)
print('Training dataset with shape', data_training.shape)
print('Batch size:', data_training.shape[2])


# Convolutional autoencoder structure using the Keras Model API
# define input shape
input_img = Input(shape=(2584,128,1))
print('Some information about tensor expected shapes')
print('Input tensor shape:', input_img.shape)

# define encoder convnet
x = Conv2D(filters=128,kernel_size=(4,4),activation='relu',padding='same')(input_img)
x = MaxPooling2D(pool_size=(2,2),strides=(2),padding='same')(x)
x = Conv2D(filters=256,kernel_size=(4,4),activation='relu',padding='same')(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2),padding='same')(x)
x = Conv2D(filters=512,kernel_size=(4,4),activation='relu',padding='same')(x)
encoded = MaxPooling2D(pool_size=(2,2),strides=(2),padding='same')(x)
print('Encoded representation tensor shape:', encoded.shape)

# define decoder convnet
x = Conv2D(filters=512,kernel_size=(4,4),activation='relu',padding='same')(encoded)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(filters=256,kernel_size=(4,4),activation='relu',padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(filters=128,kernel_size=(4,4),activation='relu',padding='same')(x)
x = UpSampling2D(size=(2,2))(x)
decoded = Conv2D(filters=1,kernel_size=(4,4),activation='linear',padding='same')(x)
print('Decoded representation tensor shape:', decoded.shape)

# define overal autoencoder model
cae = Model(inputs=input_img, outputs=decoded)
cae.compile(optimizer='adam', loss='mse')

# check for equal size
# obs: the missing value is the batch_size
if input_img.shape[1:] != decoded.shape[1:]: print('alert: in/out dimension mismatch')


# reshape the training set in 4-dimension tensor
data_training = np.reshape(data_training, (len(data_training), data_training.shape[1], data_training.shape[2], 1))
print('Data training reshaping in tensor of shape', data_training.shape)


# Autoencoder training
cae.fit(data_training,data_training,
        epochs=1,
        batch_size=4, # minibatch of 4 for memory optimisation
        #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
       )


# Saving the weights
import h5py
cae.save('cae_test.h5')

# Saving the model architecture
cae_struct = cae.to_json()

