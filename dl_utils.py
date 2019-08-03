"""DL Utilities for TensorFlow with Keras

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display as ipythondisplay
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys
from PIL import Image

def save_image_batch(dataset, DATA_DIR="./data"):
  if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
  n_img = len(dataset)
  print(f"Saving {n_img} images: ")
  for i in range(n_img):
    img_name = f"./data/image_{i+1}.png"
    print(f"Saving image {img_name} [{i+1}/{n_img}]...")
    Image.fromarray(dataset[i][:,:,::-1]).save(img_name, format='png')
  print("Saving Complete!\n")

def save_image_batch_cv2(dataset, DATA_DIR="./data"):
  if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
  n_img = len(dataset)
  print(f"Saving {n_img} images: ")
  for i in range(n_img):
    img_name = f"./data/image_cv2_{i+1}.png"
    print(f"Saving image {img_name} [{i+1}/{n_img}]...")
    cv2.imwrite(img_name, dataset[i])
  print("Saving Complete!")

def save_keras_model(model, filename="model", dirname="."):
  # Create timestamp
  from datetime import datetime
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # Save model to yaml
  model_yaml = model.to_yaml()
  yaml_filepath = os.path.join(dirname, filename + ".yaml")
  with open(yaml_filepath, "w") as yaml_file:
    yaml_file.write(model_yaml)
  print("Saved yaml file: {}".format(yaml_filepath))

  # Save weights to HDF5
  weights_filepath = os.path.join(dirname, filename + ".h5")
  model.save_weights(weights_filepath)
  print("Saved weights file: {}".format(weights_filepath))

def display_model(model):
  tf.keras.utils.plot_model(model, to_file='tmp.png', show_shapes=True)
  from IPython.display import Image
  return Image('tmp.png')

# Plotting Functions

def plot_progress(batch_org, batch_in, batch_out, figsize=(15, 5)):
  # Create figure
  plt.figure(figsize=figsize)
  
  # Diff Image (In - Original)
  diff_in = np.array(batch_in[0] - batch_org[0]).mean(axis=-1)
  diff_in = (diff_in + 1) / 2
  plt.subplot(1, 5, 1)
  plt.imshow(diff_in, cmap='gray')
  plt.title("Diff (Input - Original)")
  plt.grid(False)
  
  # Input image
  plt.subplot(1, 5, 2)
  plt.imshow(batch_in[0])
  plt.title("Input")
  plt.grid(False)
  
  # Original Image
  plt.subplot(1, 5, 3)
  plt.imshow(batch_org[0])
  plt.title("Original")
  plt.grid(False)
  
  # Output Image
  plt.subplot(1, 5, 4)
  plt.imshow(batch_out[0])
  plt.title("Output")
  plt.grid(False)
  
  # Diff Image (Out - Original)
  diff_out = np.array(batch_out[0] - batch_org[0]).mean(axis=-1)
  diff_out = (diff_out + 1) / 2
  plt.subplot(1, 5, 5)
  plt.imshow(diff_out, cmap='gray')
  plt.title("Diff (Output - Original)")
  plt.grid(False)
  
  plt.show()


def plot_loss_acc(model, target_acc=0.9, title=None):
  """
  Takes a deep learning model and plots the loss ans accuracy over epochs
  Users can supply a title if needed
  target_acc: The desired/ target acc. This parameter is needed for this function to show a horizontal bar.
  """
   
  val = True
  epochs = np.array(model.history.epoch)+1 # Add one to the list of epochs which is zero-indexed
  keys = model.history.history.keys()
  n_keys = len(model.history.history.keys())
  
  # Create Figure
  nrows = 1
  ncols = n_keys // 2 if val else n_keys
  fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 4))
  fig.subplots_adjust(wspace=.75)
  if title:
    fig.suptitle(title)
      
  for i, key in enumerate(keys):
    metric = np.array(model.history.history.get(key))
    if val:
      val_i = i + n_keys // 2
      val_key = "val_" + key
      val_metric = np.array(model.history.history.get(val_key))
      if i == ncols:
        break

    ax[i] = plt.subplot(nrows, ncols, i+1)
    color = 'tab:red'
    ax[i].set_xlabel('Epochs', fontsize=15)
    ax[i].set_ylabel(key, color=color, fontsize=15)
    ax[i].plot(epochs, metric, color=color, lw=2)
    if val:
      ax[i].plot(epochs, val_metric, color=color, lw=2, linestyle='dashed')
      plt.legend(['train', 'validate'], loc='lower left')
    ax[i].tick_params(axis='y', labelcolor=color)
    ax[i].grid(True)
    #ax[i].title.set_text(key)
    ax[i].set_title(key, fontsize=15)
  plt.show()

# plot_loss_acc(rednet_binary, title="Title")

def myplot(img_batch, autoscale=False):
  n = 10
  plt.figure(figsize=(20, 2))
  for i in range(n):
    ax = plt.subplot(1, n + 1, i + 1)
    img = np.array(img_batch[i])
    if autoscale:
      img_gray = np.array(img).mean(axis=-1)
      img_mean = img_gray.mean()
      img = (img - img_gray.min()) / img_gray.ptp()
      img = np.clip(img, 0, 1)
    plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()


def myplot_history(model):
  history = model.history.history

  # n_metrics = len()
  plt.figure(figsize=(10, 2))
  ax = plt.subplot(1, 2, 1)
  # summarize history for accuracy
  plt.plot(history['acc'])
  plt.plot(history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  
  # summarize history for loss
  ax2 = plt.subplot(1, 2, 2)
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  
  # summarize history for loss
  ax2 = plt.subplot(1, 2, 2)
  plt.plot(history['mean_absolute_error'])
  plt.plot(history['val_mean_absolute_error'])
  plt.title('model mae')
  plt.ylabel('mae')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  
  plt.show()
