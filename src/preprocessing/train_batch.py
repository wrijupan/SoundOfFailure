import numpy as np
from tensorflow import keras

sys.path += ['../src/filecheck', '../src/preprocessing',]
from filepath import *

class DataGenerator(keras.utils.Sequence):
    """Generates data batches for Tensorflow-Keras"""
    def __init__(self, wav_filepath,
                 batch_size=32,
                 dim=(32, 128),
                 hop_size=8,
                 shuffle=True):
        """
        Define DataGenerator params
        :param npy_IDs (int): Feature file (.npy)
        :param batch_size (int): Training batch size
        :param dim (tuple): Size of each data batch
        :param shuffle (bool): Shuffle indices after each epoch
        """
        self.filepath = wav_filepath
        self.batch_size = batch_size
        self.dim = dim
        self.hop_size = hop_size

        # Load features
        self.data = read_features(self.filepath)

        # Indices for total number of samples
        self.indices = np.arange(self.data.shape[0])

        # Indices for starting spectrogram sample
        self.indices_spec = np.arange(self.data.shape[1]-self.dim[0]+self.hop_size,
                                 step=self.hop_size)
        self.indexmax = len(self.indices_spec)

        self.indices = np.repeat(self.indices, self.indexmax)
        self.indices_spec = np.repeat(self.indices_spec, self.data.shape[0])

        # Shuffle indices at the beginning and end of epoch
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Shuffle indices at each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
            np.random.shuffle(self.indices_spec)

    def __len__(self):
        """
        Total number of batches per epoch
        # samples /  batch_size
        """
        n_samples = self.data.shape[0] * self.indexmax
        return int(np.floor(n_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Gets training data batches
        :param index (int): Batch index
            0 to total no. of batches
        :return: DataGenerator object
        """
        indices = self.indices[index*self.batch_size :
                               (index+1)*self.batch_size]

        indices_spec = self.indices_spec[index*self.batch_size:
                                         (index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(indices,
                                   indices_spec).reshape((self.batch_size,
                                                            *self.dim,
                                                            1))
        return X, X

    def __data_generation(self, indexes, indexes_start):
        """
        Generate one batch of data
        :param indexes:
        :param indexes_start:
        :return:
        """
        batch_data = np.empty((self.batch_size, *self.dim))

        for i, (ind, ind_start) in enumerate(zip(indexes, indexes_start)):
            x = self.data[ind,]
            length, mels = x.shape

            start = ind_start
            start = min(start, length-self.dim[0])

            batch_data[i,] = x[start:
                               start+self.dim[0],
                             :]

            return batch_data



