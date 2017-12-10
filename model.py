"""Methods relating to model and training."""

from time import time
import tensorflow as tf

from constants import N_HIDDENS, N_EPOCHS


def build_model(inputs, targets, lengths):
    """Build a simple LSTM."""
    outputs = tf.layers.dense(tf.nn.dynamic_rnn(
        cell=tf.nn.rnn_cell.LSTMCell(N_HIDDENS), inputs=inputs,
        sequence_length=lengths, dtype=tf.float32)[0], inputs.shape.as_list()[-1],
        activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(targets, outputs)
    updates = tf.train.AdamOptimizer().minimize(loss)

    return loss, updates

def train_model(train_dataset, valid_dataset):
    """Create an iterator over a dataset and train a model."""

    def run_training_loop(sess, model_ops, data_ops, n_epochs, verbose=False):
        """Run training loop."""

        def run_model(sess, loss, data_init_op, updates, verbose):
            """Run one pass of model through all data mini-batches."""
            sess.run(data_init_op)
            while True:
                try:
                    rvals = sess.run([loss, updates] if updates != None else [loss])
                    if verbose:
                        print '\tMinibatch loss: %.3f' % rvals[0]
                except tf.errors.OutOfRangeError:
                    if verbose:
                        print 'End of mini-batches.'
                    break

        loss, updates = model_ops
        train_init_op, valid_init_op = data_ops
        for i in range(n_epochs):
            epoch_start = time()

            # Training step
            if verbose:
                print 'EPOCH %d, training step'
            run_model(sess, loss, train_init_op, updates, verbose)

            # Validation step
            if verbose:
                print 'EPOCH %d, validation step'
            run_model(sess, loss, valid_init_op, None, verbose)

            print "Epoch %d time: %.5f" % (i, time()-epoch_start)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    inputs, targets, lengths = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)

    loss, updates = build_model(inputs, targets, lengths)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        run_training_loop(sess, (loss, updates), (train_init_op, valid_init_op),
                          N_EPOCHS, verbose=False)


