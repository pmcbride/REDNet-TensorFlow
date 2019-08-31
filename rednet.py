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
# tf.enable_eager_execution()
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SpatialDropout2D, ReLU
from tensorflow.keras.models import Model


import matplotlib.pyplot as plt
import numpy as np

n_layers = 5
INPUT_SHAPE = (32, 32, 3)
# INPUT_SHAPE = (64, 64, 3)

def REDNet(n_layers=5, n_skip=2, input_shape=(32, 32, 3), tensor=None, enable_skip=False, debug=False, compile=False, dropout=False):
  # Define number of deconv layers and skip connection frequency
  n_conv_layers = n_layers
  n_deconv_layers = n_layers
  n_skip = n_skip

  def _conv_block(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), conv_id=1, dropout=False):
    x = Conv2D(filters, kernel_size, strides,
                               activation='relu',
                               padding='same',
                               name=f'conv{conv_id}')(inputs)
    if dropout:
      return SpatialDropout2D(0.2)(x)
    return x

  def _deconv_block(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), deconv_id=1, dropout=False):
    x = Conv2DTranspose(filters, kernel_size, strides,
                               activation='relu',
                               padding='same',
                               name=f'deconv{deconv_id}')(inputs)
    if dropout:
      return SpatialDropout2D(0.2)(x)
    return x

  def _skip_block(input1, input2, skip_id=1):
    x = tf.keras.layers.add([input1, input2])
    return ReLU(name=f'skip{skip_id}_relu')(x)

  def _build_layer_list(model):
    model_layers = [layer for layer in model.layers]
    model_outputs = [layer.output for layer in model.layers]
    return model_layers, model_outputs

  # CREATE ENCODER MODEL
  encoder_inputs = Input(shape=input_shape, dtype='float32', name="encoder_inputs") # inputs skips to end

  for i in range(n_conv_layers):
    conv_idx = i + 1
    if conv_idx == 1:
      conv = _conv_block(encoder_inputs, conv_id=conv_idx)
    else:
      conv = _conv_block(conv, conv_id=conv_idx)

  encoded = conv
  encoder = Model(inputs=encoder_inputs, outputs=encoded, name='encoder')

  # Create encoder layer and output lists
  encoder_layers, encoder_outputs = _build_layer_list(encoder)

  # CREATE AUTOENCODER MODEL
  for i, skip in enumerate(reversed(encoder_outputs[:-1])):
    deconv_idx = i + 1
    deconv_filters = 32
    if deconv_idx == n_deconv_layers:
      deconv_filters = 3

    if deconv_idx == 1:
      #deconv = _deconv_block(decoder_inputs, filters=deconv_filters, deconv_id=deconv_idx)
      deconv = _deconv_block(encoded, filters=deconv_filters, deconv_id=deconv_idx)
      if debug: print(f"Deconv... \t deconv_idx: {deconv_idx}, filters: {deconv_filters}, shape: {deconv.shape[1:]}, input_name: '{encoded.name}', deconv_name: '{deconv.name}'")
    else:
      deconv = _deconv_block(deconv, filters=deconv_filters, deconv_id=deconv_idx)
      if debug: print(f"Deconv... \t deconv_idx: {deconv_idx}, filters: {deconv_filters}, shape: {deconv.shape[1:]}, input_name: '{deconv.name}', deconv_name: '{deconv.name}'")

    if enable_skip:
      if deconv_idx % n_skip == 0:
        skip_num = deconv_idx // n_skip
        if debug: print(f"Skip... \t deconv_idx: {deconv_idx}, skip #: {skip_num}, conv_shape: {skip.shape[1:]}, deconv_shape: {deconv.shape[1:]} \n\t\t conv: '{skip.name}', deconv: '{deconv.name}'")
        #assert deconv.shape == skip.shape
        deconv = _skip_block(deconv, skip, skip_id=skip_num)
        if debug: print(f"Added... \t deconv_idx: {deconv_idx}, filters: {deconv_filters}, shape: {deconv.shape[1:]}\n")

  decoded = deconv #(decoder_inputs)
  model = Model(inputs=encoder_inputs, outputs=decoded, name=f'REDNet{n_conv_layers}')

  # Create model layer and output lists
  model_layers, model_outputs = _build_layer_list(model)

  # CREATE DECODER MODEL
  encoded_input = Input(shape=encoded.shape[1:])
  decoder_layer = model.layers[-1]
  decoder = Model(encoded_input, decoder_layer(encoded_input))

  # Create decoder layer and output lists
  decoder_layers, decoder_outputs = _build_layer_list(decoder)

  if debug:
    for i, out in enumerate(encoder_outputs):
      print(i, out)
    print()
    for i, out in enumerate(decoder_layers):
      print(i, out)
    print()
    for i, out in enumerate(model_outputs):
      print(i, out)
    print()

  if compile:
    # Compile Model and fit
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    model.summary()
  return model, encoder, decoder

if __name__ == '__main__':
  import os
  import h5py
  model_dir = "D:\OneDrive\Code\REDNet-TensorFlow\models"
  model_name = "rednet_model.h5"
  weights_name = "rednet_weights_5c5d2s.h5"
  model_path = os.path.join(model_dir, model_name)
  weights_path = os.path.join(model_dir, weights_name)
  model, encoder, decoder = REDNet()
  model.load_weights(weights_path)
  print("Loaded models:")
  model.summary()
