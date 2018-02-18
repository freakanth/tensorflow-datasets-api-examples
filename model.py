"""Methods relating to model and training."""

from time import time
import tensorflow as tf

from constants import N_HIDDENS, N_EPOCHS


def build_model(inputs, targets, lengths):
    """Build a simple LSTM."""

    outputs = tf.layers.dense(tf.nn.dynamic_rnn(
        cell=tf.nn.rnn_cell.LSTMCell(N_HIDDENS), inputs=inputs, sequence_length=lengths,
        dtype=tf.float32)[0], inputs.shape.as_list()[-1], activation=tf.nn.sigmoid)
    loss = tf.losses.mean_squared_error(targets, outputs)
    updates = tf.train.AdamOptimizer().minimize(loss)

    return loss, updates

def train_model(train_dataset, valid_dataset):
    """Create an iterator over a dataset and train a model."""
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
        run_training_loop(sess, (loss, updates), (train_init_op, valid_init_op), N_EPOCHS)
#        run_data_loop(sess, (train_init_op, valid_init_op, (inputs, targets, lengths)),
#                      N_EPOCHS, verbose=False)

def run_training_loop(sess, model_ops, data_ops, n_epochs):
    """Run training loop."""

    def run_model(sess, (loss, updates), data_init_op):
        """Run one pass of model through all data mini-batches."""
        sess.run([data_init_op]); losses = []
        while True:
            try:
                losses.append(sess.run([loss, updates] if updates != None else [loss])[0])
            except tf.errors.OutOfRangeError:
                break
        return sum(losses)/len(losses)

    loss, updates = model_ops
    train_init_op, valid_init_op = data_ops
    for i in range(n_epochs):
        epoch_start = time()

        # Training step (with updates)
        train_loss = run_model(sess, (loss, updates), train_init_op)

        # Validation step (without updates) 
        valid_loss = run_model(sess, (loss, None), valid_init_op)

        print("Epoch %d time: %.5f, training loss: %.3f, validation loss: %.3f"
              % (i, time()-epoch_start, train_loss, valid_loss))

def run_data_loop(sess, data_ops, n_epochs):
    """Run data retrieval loop."""

    def retrieve_data(sess, next_item, data_init_op):
        """Retrieve mini-batches."""
        sess.run(data_init_op)
        data = []
        while True:
            try:
                data.append(sess.run(next_item))
            except tf.errors.OutOfRangeError:
                break
        return data

    train_init_op, valid_init_op, next_item = data_ops
    for i in range(n_epochs):
        epoch_start = time()

        # Training step (with updates)
        retrieve_data(sess, next_item, train_init_op)

        # Validation step (without updates)
        retrieve_data(sess, next_item, valid_init_op)

        print "Epoch %d time: %.5f" % (i, time()-epoch_start)


