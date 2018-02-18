"""Script to experiment with tf.data"""

import sys
from time import time
import tensorflow as tf

from data import make_synthetic_batched_data, batched_generator_dataset
from model import build_model
from constants import BATCH_SIZE, MAX_SEQ_LEN, N_DIMS, N_TRAIN, N_VALID, N_EPOCHS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To reduce exception verbosity

def with_placeholders():
    """Train a model with placeholder input."""

    train_data = make_synthetic_batched_data(N_TRAIN, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)
    valid_data = make_synthetic_batched_data(N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SEQ_LEN, N_DIMS])
    targets = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SEQ_LEN, N_DIMS])
    lengths = tf.placeholder(dtype=tf.int32, shape=[None])

    loss, updates = build_model(inputs, targets, lengths)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(N_EPOCHS):
            epoch_start = time()

            for (mb_ins, mb_outs, mb_lens) in zip(*train_data):
                minibatch_loss, _ = sess.run(
                    (loss, updates), feed_dict={inputs: mb_ins,
                                                targets: mb_outs,
                                                lengths: mb_lens})

            for (mb_ins, mb_outs, mb_lens) in zip(*valid_data):
                minibatch_loss = sess.run(
                    loss, feed_dict={inputs: mb_ins,
                                     targets: mb_outs,
                                     lengths: mb_lens})

            print("Epoch %d time: %.5f" % (i, time()-epoch_start))

def with_initialisable_iterators():
    """Train a model with initialisable iterator input."""

    train_dataset = batched_generator_dataset(N_TRAIN, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)
    train_dataset.shuffle(buffer_size=N_TRAIN)
    valid_dataset = batched_generator_dataset(N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    inputs, targets, lengths = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)

    loss, updates = build_model(inputs, targets, lengths)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(N_EPOCHS):
            epoch_start = time()

            # Training method 2
            sess.run(train_init_op)
            while True:
                try:
                    minibatch_loss, _ = sess.run((loss, updates))
                except tf.errors.OutOfRangeError:
                    break

            sess.run(valid_init_op)
            while True:
                try:
                    minibatch_loss = sess.run(loss)
                except tf.errors.OutOfRangeError:
                    break

            print("Epoch %d time: %.5f" % (i, time()-epoch_start))

def with_feedable_iterators():
    """Train a model with feedable iterator input."""

    train_dataset = batched_generator_dataset(N_TRAIN, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)
    train_dataset.shuffle(buffer_size=N_TRAIN)
    valid_dataset = batched_generator_dataset(N_VALID, BATCH_SIZE, MAX_SEQ_LEN, N_DIMS)

    train_iterator = train_dataset.make_initializable_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types, train_iterator.output_shapes)
    inputs, targets, lengths = iterator.get_next()

    loss, updates = build_model(inputs, targets, lengths)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_iterator_handle = sess.run(train_iterator.string_handle())
        valid_iterator_handle = sess.run(valid_iterator.string_handle())
        for i in range(N_EPOCHS):
            epoch_start = time()

            # Training method 3
            sess.run(train_iterator.initializer)
            while True:
                try:
                    minibatch_loss, _ = sess.run(
                        (loss, updates), feed_dict={handle: train_iterator_handle})
                except tf.errors.OutOfRangeError:
                    break

            sess.run(valid_iterator.initializer)
            while True:
                try:
                    minibatch_loss = sess.run(
                        loss, feed_dict={handle: valid_iterator_handle})
                except tf.errors.OutOfRangeError:
                    break

            print("Epoch %d time: %.5f" % (i, time()-epoch_start))

METHOD_DICT = {
    '1': with_placeholders,
    '2': with_initialisable_iterators,
    '3': with_feedable_iterators
}

if __name__ == '__main__':
    METHOD_DICT[sys.argv[1]]()
