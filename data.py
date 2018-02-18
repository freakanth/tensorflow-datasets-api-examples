"""Methods relating to data creataion and processing."""

from functools import partial
import tempfile
import numpy as np
import tensorflow as tf


def apply_pipeline(dataset, opts):
    """Apply the steps in a data pipeline to the dataset."""

    def _cache(dataset):
        return dataset.cache()

    def _shuffle(buffer_size, dataset):
        return dataset.shuffle(buffer_size)

    def _prefetch(n_items, dataset):
        return dataset.prefetch(n_items)

    def _padded_batch(batch_size, n_dims, dataset):
        return dataset.padded_batch(
            batch_size, ((None, n_dims), (None, n_dims), ())) if batch_size > 0 else dataset

    return _cache(
        _shuffle(
            opts['n_batches'],
            _padded_batch(
                opts['batch_size'], opts['n_dims'],
                _prefetch(
                    opts['n_items']/2,
                    dataset))))

def generator(items):
    """Generator function to use with tf.data.Dataset.from_generator."""
    for item in zip(*items):
        yield item

def batched_generator_dataset(n_items, batch_size, max_seq_len, n_dims):
    """Make a tf.data.Dataset object via a callable Numpy 3D ndarray generator."""

    def make_synthetic_batched_data(n_items, batch_size, seq_len, n_dims):
        """Create a synthetic dataset."""

        def pad_and_batch(data, seq_len, batch_size):
            """Batch unbatched sequence data."""

            def _pad(seq, n_zero):
                """Pad a vector sequence with n zero-vectors."""
                return np.concatenate((seq, np.zeros((n_zero, seq.shape[-1]))), axis=0)

            ins, outs, lens = data
            pad_ins = [_pad(ins[i], seq_len-lens[i]) for i in xrange(len(ins))]
            pad_outs = [_pad(outs[i], seq_len-lens[i]) for i in xrange(len(outs))]

            return ([np.stack(pad_ins[i:min(i+batch_size, len(pad_ins))])
                     for i in xrange(0, len(pad_ins), batch_size)],
                    [np.stack(pad_outs[i:min(i+batch_size, len(pad_outs))])
                     for i in xrange(0, len(pad_outs), batch_size)],
                    [np.stack(lens[i:min(i+batch_size, len(lens))])
                     for i in xrange(0, len(lens), batch_size)])

        return pad_and_batch(
            make_synthetic_unbatched_data(n_items, seq_len, n_dims), seq_len, batch_size)

    data = make_synthetic_batched_data(n_items, batch_size, max_seq_len, n_dims)
    return apply_pipeline(
        dataset=tf.data.Dataset.from_generator(
            partial(generator, data), (tf.float32, tf.float32, tf.int32),
            (tf.TensorShape([None, max_seq_len, n_dims]),
             tf.TensorShape([None, max_seq_len, n_dims]),
             tf.TensorShape([None,]))),
        opts={'batch_size': 0, 'n_items': n_items,
              'n_batches': n_items//batch_size + int(n_items % batch_size > 0),
              'n_dims': n_dims})

def make_synthetic_unbatched_data(n_items, seq_len, n_dims):
    """Create a synthetic dataset."""
    np.random.seed(0xbeef)
    lens = [np.random.randint(seq_len//2, seq_len) for i in range(n_items)]

    return ([np.random.randn(lens[i], n_dims) for i in range(n_items)],
            [np.random.randn(lens[i], n_dims) for i in range(n_items)], lens)

def unbatched_generator_dataset(n_items, batch_size, max_seq_len, n_dims):
    """Make a tf.data.Dataset object via a callable Numpy 2D ndarray generator."""

    data = make_synthetic_unbatched_data(n_items, max_seq_len, n_dims)
    return apply_pipeline(
        dataset=tf.data.Dataset.from_generator(
            partial(generator, data), (tf.float32, tf.float32, tf.int32),
            (tf.TensorShape([None, n_dims]), tf.TensorShape([None, n_dims]), tf.TensorShape([]))),
        opts={'batch_size': batch_size, 'n_items': n_items,
              'n_batches': n_items//batch_size + int(n_items % batch_size > 0),
              'n_dims': n_dims})

def tfrecord_dataset(n_items, batch_size, max_seq_len, n_dims):
    """Make a tf.data.Dataset object via tf.data.TFRecordDataset()."""

    def write_to_tfrecord_file((inputs, targets, lengths), name):
        """Write all examples into a TFRecords file"""

        def make_sequence_example(ins, tgts, lens):
            """Make an instance of tf.train.SequenceExample."""
            return tf.train.SequenceExample(
                context=tf.train.Features(
                    feature={"length": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[lens]))}),
                feature_lists=tf.train.FeatureLists(
                    feature_list=dict([('inputs', tf.train.FeatureList(
                        feature=[tf.train.Feature(
                            float_list=tf.train.FloatList(value=ins[t, :]))
                                 for t in range(lens)])),
                                       ('targets', tf.train.FeatureList(
                        feature=[tf.train.Feature(
                            float_list=tf.train.FloatList(value=tgts[t, :]))
                                 for t in range(lens)]))])))

        with tf.python_io.TFRecordWriter(open(name, 'wb').name) as writer:
            map(lambda item: writer.write(make_sequence_example(*item).SerializeToString()),
                zip(inputs, targets, lengths))
            print "Wrote TFRecord to %s" % name

    def tfrecord_parse_function(n_dims, serialised):
        """Parse a single sequence example in the SequenceExample protocol buffer."""

        context_features, sequence_features = tf.parse_single_sequence_example(
            serialized=serialised,
            context_features={"length": tf.FixedLenFeature((), dtype=tf.int64)},
            sequence_features={"inputs": tf.FixedLenSequenceFeature((n_dims), dtype=tf.float32),
                               "targets": tf.FixedLenSequenceFeature((n_dims), dtype=tf.float32)})

        return (sequence_features['inputs'], sequence_features['targets'],
                context_features['length'])

    data = make_synthetic_unbatched_data(n_items, max_seq_len, n_dims)
    name = tempfile.mktemp(suffix='.tfr', dir='.')
    write_to_tfrecord_file(data, name)
    dataset = tf.data.TFRecordDataset(name).map(partial(tfrecord_parse_function, n_dims))
    return apply_pipeline(dataset=dataset,
                          opts={'batch_size': batch_size, 'n_items': n_items,
                                'n_batches': n_items//batch_size + int(n_items % batch_size > 0),
                                'n_dims': n_dims})
