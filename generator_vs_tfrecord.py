"""Some examples of creating sequence datasets using the Tensorflow Datasets API."""

import os
import sys
import glob

from constants import N_TRAIN, N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS
from data import unbatched_generator_dataset, batched_generator_dataset, tfrecord_dataset
from model import train_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To reduce exception verbosity

def read_unbatched_ndarrays(n_train, n_valid, batch_size, max_seq_len, n_dims):
    """Training a model with Dataset created using a generator of unbatched data."""

    train_model(unbatched_generator_dataset(n_train, batch_size, max_seq_len, n_dims),
                unbatched_generator_dataset(n_valid, batch_size, max_seq_len, n_dims))

def read_batched_ndarrays(n_train, n_valid, batch_size, max_seq_len, n_dims):
    """Training a model with Dataset created using a generator of batched data."""

    train_model(batched_generator_dataset(n_train, batch_size, max_seq_len, n_dims),
                batched_generator_dataset(n_valid, batch_size, max_seq_len, n_dims))

def read_unbatched_tfrecord(n_train, n_valid, batch_size, max_seq_len, n_dims):
    """Training a model with Dataset created using a TFRecord file."""

    map(os.remove, glob.glob('*.tfr')) #  Remove any existing TFRecord files
    train_model(tfrecord_dataset(n_train, batch_size, max_seq_len, n_dims),
                tfrecord_dataset(n_valid, batch_size, max_seq_len, n_dims))

METHOD = {
    '1': read_unbatched_ndarrays,
    '2': read_batched_ndarrays,
    '3': read_unbatched_tfrecord
}

if __name__ == '__main__':
    METHOD_ID = sys.argv[1]
    METHOD[METHOD_ID](N_TRAIN, N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)
