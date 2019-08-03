"""REDNet model for TensorFlow with Keras

# Reference
- [Image Restoration Using Convolutional Auto-encoders
   with Symmetric Skip Connections](https://arxiv.org/abs/1704.04861)

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

KERNEL_SIZE = (3, 3)
FILTER_NUMBER = 32 # base number of convolutional filters
STRIDES = (1, 1)
INPUT_SHAPE = (64, 64, 3)
n_outputs = 1 # number of outputs (i.e., face or not face)


def REDNet(input_shape=(64, 64, 3), final_activation='relu'):
  """
  Define the encoder network for the REDNet Autoencoder
  """
  params = dict(kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                use_bias=True)
  inputs = layers.Input(shape=input_shape)  # inputs skips to end
  conv1 = _conv_block(inputs, conv_id=1)
  conv2 = _conv_block(conv1, conv_id=2)  # conv2 skips to deconv8
  conv3 = _conv_block(conv2, conv_id=3)
  conv4 = _conv_block(conv3, conv_id=4)  # conv4 skips to deconv6
  conv5 = _conv_block(conv4, conv_id=5)
  conv6 = _conv_block(conv5, conv_id=6)  # conv6 skips to deconv4
  conv7 = _conv_block(conv6, conv_id=7)
  conv8 = _conv_block(conv7, conv_id=8)  # conv8 skips to deconv2
  conv9 = _conv_block(conv8, conv_id=9)
  conv10 = _conv_block(conv9, conv_id=10)
  encoded = conv10
  deconv1 = _deconv_block(encoded, deconv_id=1)
  deconv2 = _deconv_block(deconv1, deconv_id=2)
  skip1 = _skip_block(conv8, deconv2, skip_id=1)  # skip1
  deconv3 = _deconv_block(skip1, deconv_id=3)
  deconv4 = _deconv_block(deconv3, deconv_id=4)
  skip2 = _skip_block(conv6, deconv4, skip_id=2)  # skip2
  deconv5 = _deconv_block(skip2, deconv_id=5)
  deconv6 = _deconv_block(deconv5, deconv_id=6)
  skip3 = _skip_block(conv4, deconv6, skip_id=3)  # skip3
  deconv7 = _deconv_block(skip3, deconv_id=7)
  deconv8 = _deconv_block(deconv7, deconv_id=8)
  skip4 = _skip_block(conv2, deconv8, skip_id=4)  # skip4
  deconv9 = _deconv_block(skip4, deconv_id=9)
  deconv10 = _deconv_block(deconv9, filters=3, deconv_id=10)
  #deconv10_flatten = layers.Conv2D(3, (1, 1), padding='same', name="deconv10_flatten")(deconv10)
  skip5 = _skip_block(inputs, deconv10, skip_id=5) # skip5
  #skip5 = tf.add(inputs, deconv10_flatten)
  
  if final_activation == 'sigmoid':
    decoded = layers.Activation('sigmoid', name="sigmoid_decoded_layer")(skip5)
  elif final_activation == 'relu':
    decoded = skip5
    #decoded = layers.ReLU(name="relu_decoded_layer")(skip5)
  elif final_activation == 'relu_1':
    decoded = layers.ReLU(1, name="relu_1_decoded_layer")(skip5)
  
  print("decoded: ", decoded)
  # finalize the model
  model = tf.keras.Model(inputs=inputs, outputs=decoded, name='REDNet10')
  
  return model

def _conv_block(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), conv_id=1):
  x = layers.Conv2D(filters, kernel_size, strides,
                    activation='relu',  # depth_multiplier=3,
                    padding='same',
                    name=f'conv{conv_id}')(inputs)
  return x

def _deconv_block(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), deconv_id=1):
  x = layers.Conv2DTranspose(filters, kernel_size, strides,
                             activation='relu',  # depth_multiplier=3,
                             padding='same',
                             name=f'deconv{deconv_id}')(inputs)
  return x

def _skip_block(input1, input2, skip_id=1):
  x = tf.add(input1, input2)
  return layers.ReLU(name=f'skip{skip_id}_relu')(x)