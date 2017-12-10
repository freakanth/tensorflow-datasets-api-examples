"""Some examples of creating sequence datasets using the Tensorflow Datasets API."""

import sys

from constants import N_TRAIN, N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS
from data import unbatched_generator_dataset, batched_generator_dataset, tfrecord_dataset
from model import train_model

TRAIN_RECORD_NAME = 'training_sequences.tfr'
VALID_RECORD_NAME = 'validation_sequences.tfr'


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

    train_model(tfrecord_dataset(n_train, batch_size, max_seq_len, n_dims, TRAIN_RECORD_NAME),
                tfrecord_dataset(n_valid, batch_size, max_seq_len, n_dims, VALID_RECORD_NAME))

METHOD = {
    '1': read_unbatched_ndarrays,
    '2': read_batched_ndarrays,
    '3': read_unbatched_tfrecord
}

if __name__ == '__main__':
    METHOD_ID = sys.argv[1]
    METHOD[METHOD_ID](N_TRAIN, N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)
