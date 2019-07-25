import numpy as np
import tensorflow
from tensorflow.keras.utils import HDF5Matrix
import numpy as np
import h5py
import sys

## HDF5 Dataset Loader
class HDF5DatasetLoader(object):
  def __init__(self, data_path):

    print("Opening {}".format(data_path))
    sys.stdout.flush()

    print("Loading data...")
    sys.stdout.flush()
    
    self.images = HDF5Matrix(data_path, 'images')
    self.labels = HDF5Matrix(data_path, 'labels')
    self.image_dims = self.images.shape
    n_train_samples = self.image_dims[0]

    self.train_inds = np.arange(n_train_samples)

    face_labels = (self.labels[:] == 1).reshape(n_train_samples,)
    notface_labels = (self.labels[:] != 1).reshape(n_train_samples,)
    self.face_idx = self.train_inds[face_labels]
    self.notface_idx = self.train_inds[notface_labels]

    print("Data Loaded...")
    sys.stdout.flush()

  def get_train_size(self):
    return self.train_inds.shape[0]

  def get_train_steps_per_epoch(self, batch_size):#, factor=10):
    return self.get_train_size()//batch_size#//factor

  def get_batch(self, n, only_faces=False, return_inds=False):
    if only_faces:
      selected_inds = self.face_idx #np.random.choice(self.face_idx, size=n, replace=False)#, p=p_pos)
    else:
      n_split = n // 2
      selected_pos_inds = self.face_idx[:n_split] #np.random.choice(self.face_idx, size=n//2, replace=False)#, p=p_pos)
      selected_neg_inds = self.notface_idx[:n_split] #np.random.choice(self.notface_idx, size=n//2, replace=False)#, p=p_neg)
      selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

    sorted_inds = np.sort(selected_inds)
    train_img = self.images[sorted_inds]
    train_img = train_img[...,::-1]
    train_label = self.labels[sorted_inds][...]
    return (train_img, train_label, sorted_inds) if return_inds else (train_img, train_label)

  def get_all_train_faces(self):
    return self.images[self.face_idx][..., ::-1]


## Training Dataset Loader
class TrainingDatasetLoader(object):
  def __init__(self, data_path):

    print("Opening {}".format(data_path))
    sys.stdout.flush()

    self.cache = h5py.File(data_path, 'r')
    
    print("Loading data into memory...")
    sys.stdout.flush()
    
    self.images = self.cache['images'][:]
    self.labels = self.cache['labels'][:]
    self.image_dims = self.images.shape
    n_train_samples = self.image_dims[0]

    self.train_inds = np.random.permutation(np.arange(n_train_samples))

    self.face_idx = self.train_inds[ self.labels[self.train_inds, 0] == 1.0 ]
    self.not_face_idx = self.train_inds[ self.labels[self.train_inds, 0] != 1.0 ]

  def get_train_size(self):
    return self.train_inds.shape[0]

  def get_train_steps_per_epoch(self, batch_size, factor=10):
    return self.get_train_size()//factor//batch_size

  def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
    if only_faces:
      selected_inds = np.random.choice(self.face_idx, size=n, replace=False, p=p_pos)
    else:
      selected_pos_inds = np.random.choice(self.face_idx, size=n//2, replace=False, p=p_pos)
      selected_neg_inds = np.random.choice(self.not_face_idx, size=n//2, replace=False, p=p_neg)
      selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

    sorted_inds = np.sort(selected_inds)
    train_img = self.images[sorted_inds,:,:,::-1]#/255.
    train_label = self.labels[sorted_inds,...]
    return (train_img, train_label, sorted_inds) if return_inds else (train_img, train_label)

  def get_n_most_prob_faces(self, prob, n):
    idx = np.argsort(prob)[::-1]
    most_prob_inds = self.face_idx[idx[:10*n:10]]
    return self.images[most_prob_inds,...]/255.

  def get_all_train_faces(self):
    return self.images[ self.face_idx, :, :, ::-1 ]