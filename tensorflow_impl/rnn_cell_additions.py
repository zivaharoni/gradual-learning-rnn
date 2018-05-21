import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

core_rnn_cell = tf.contrib.rnn


class VariationalDropoutWrapper(core_rnn_cell.RNNCell):
    def __init__(self, cell, batch_size, hidden_size, output_keep_prob=1.0, state_keep_prob=1.0):
        self._cell = cell
        self._new_output_keep_prob = tf.placeholder(tf.float32, shape=[], name="output_keep_prob")
        self._new_state_keep_prob = tf.placeholder(tf.float32, shape=[], name="state_keep_prob")
        self._output_keep_prob = tf.Variable(output_keep_prob, trainable=False)
        self._state_keep_prob = tf.Variable(state_keep_prob, trainable=False)

        self._output_keep_prob_update = tf.assign(self._output_keep_prob, self._new_output_keep_prob)
        self._state_keep_prob_update = tf.assign(self._state_keep_prob, self._new_state_keep_prob)

        self._batch_size = batch_size
        with tf.name_scope("variational_masks"):
            self._output_mask = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, hidden_size], name="output_mask")
            self._state_mask = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, hidden_size], name="state_mask")

            with tf.name_scope("out_mask_gen"):
                random_tensor = ops.convert_to_tensor(self._output_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size], seed=570164)
                self._gen_output_mask = math_ops.floor(random_tensor)

            with tf.name_scope("rec_mask_gen"):
                random_tensor = ops.convert_to_tensor(self._state_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.state_size.h], seed=570164)
                self._gen_state_mask = math_ops.floor(random_tensor)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def gen_masks(self, session):
            return {self._output_mask: session.run(self._gen_output_mask), self._state_mask: session.run(self._gen_state_mask)}  # , self._assign_mem_mask

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        session.run(self._output_keep_prob_update, feed_dict={self._new_output_keep_prob: output_keep_prob})
        session.run(self._state_keep_prob_update, feed_dict={self._new_state_keep_prob: state_keep_prob})

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        output, state = self._cell(inputs, state, scope)
        (c, h) = state
        with tf.name_scope("Dropout"):
            if (isinstance(self._output_keep_prob, float) and
                        self._output_keep_prob == 1):
                return
            new_h_out = math_ops.div(h, self._output_keep_prob) * self._output_mask
            new_h_fb = math_ops.div(h, self._state_keep_prob) * self._state_mask
            new_state = core_rnn_cell.LSTMStateTuple(c, new_h_fb)

        return new_h_out, new_state


class WeightDroppedVariationalDropoutWrapper(core_rnn_cell.RNNCell):
    def __init__(self, cell, batch_size, hidden_size, drop_hid=False):
        self._cell = cell
        self._new_output_keep_prob = tf.placeholder(tf.float32, shape=[], name="output_keep_prob")
        self._output_keep_prob = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="output_keep_prob")
        self._output_keep_prob_update = tf.assign(self._output_keep_prob, self._new_output_keep_prob,
                                                  name="update_output_mask")
        self._drop_hid = drop_hid

        if drop_hid:
            self._new_state_keep_prob = tf.placeholder(tf.float32, shape=[], name="state_keep_prob")
            self._state_keep_prob = tf.Variable(0.0, dtype=tf.float32, trainable=False)
            self._state_keep_prob_update = tf.assign(self._state_keep_prob, self._new_state_keep_prob)

        self._batch_size = batch_size
        with tf.name_scope("variational_masks"):
            self._output_mask = tf.placeholder(dtype=tf.float32,
                                               shape=[self._batch_size, hidden_size],
                                               name="output_mask")

            with tf.name_scope("out_mask_gen"):
                random_tensor = ops.convert_to_tensor(self._output_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size],seed=570164, dtype=tf.float32)
                self._gen_output_mask = math_ops.floor(random_tensor)

            if drop_hid:
                self._state_mask = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, hidden_size],
                                                  name="output_mask")
                with tf.name_scope("rec_mask_gen"):
                    random_tensor = ops.convert_to_tensor(self._state_keep_prob)
                    random_tensor += random_ops.random_uniform([self._batch_size, self._cell.state_size.h], seed=570164, dtype=tf.float32)
                    self._gen_state_mask = math_ops.floor(random_tensor)

    @property
    def cell(self):
        return self._cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def gen_masks(self, session):
        masks = session.run({self._output_mask: self._gen_output_mask})
        if self._drop_hid:
            masks.update(session.run({self._state_mask: self._gen_state_mask}))

        # masks.update(self._cell.gen_masks(session))

        return masks

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        session.run(self._output_keep_prob_update,
                    feed_dict={self._new_output_keep_prob: output_keep_prob})
        if self._drop_hid:
            session.run(self._state_keep_prob_update,
                        feed_dict={self._new_state_keep_prob: state_keep_prob})
        self._cell.update_drop_params(session, state_keep_prob)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        output, state = self._cell(inputs, state, scope)
        (c, h) = state
        with tf.name_scope("drop_cell_outputs"):
            if (isinstance(self._output_keep_prob, float) and
                        self._output_keep_prob == 1):
                return
            new_h_out = math_ops.div(h, self._output_keep_prob) * self._output_mask
            if self._drop_hid:
                h = math_ops.div(h, self._state_keep_prob) * self._state_mask
            new_state = core_rnn_cell.LSTMStateTuple(c, h)

        return new_h_out, new_state


class WeightDroppedLSTMCell(BasicLSTMCell):
    def __init__(self, num_units, is_training, dtype=tf.float32, state_is_tuple=True, reuse=None):

        super(BasicLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._is_training = is_training
        self._num_units = num_units
        self._dtype = dtype
        self._state_is_tuple = state_is_tuple

        self._new_rec_keep_prob = tf.placeholder(tf.float32, shape=[], name="new_rec_keep_prob")
        self._rec_keep_prob = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="weights_keep_prob")

        self._rec_keep_prob_update = tf.assign(self._rec_keep_prob, self._new_rec_keep_prob, name="update_rec_mask")

        with tf.name_scope("DropConnect_masks"):
            self._rec_mask = tf.placeholder(dtype=tf.float32,
                                            shape=[num_units, 4 * num_units],
                                            name="recursive_mask")

            with tf.name_scope("rec_mask_gen"):
                random_tensor = ops.convert_to_tensor(self._rec_keep_prob)
                random_tensor += random_ops.random_uniform([num_units, 4 * num_units], seed=570164, dtype=tf.float32)
                self._gen_rec_mask = math_ops.floor(random_tensor)

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell Batch Normalized(LSTM)."""
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            raise ValueError("no compatible with state is NOT a tuple")

        with tf.name_scope("lstm_weights"):
            wx = tf.get_variable("Wx",
                                 shape=[inputs.shape.as_list()[-1], 4 * self._num_units],
                                 dtype=self._dtype,)
            wh = tf.get_variable("Wh",
                                 shape=[self._num_units, 4 * self._num_units],
                                 dtype=self._dtype)
            b = tf.get_variable("b", shape=[4 * self._num_units], dtype=self._dtype)

        with tf.name_scope("projection"):
            wx_x = tf.matmul(inputs, wx)
            if self._is_training:
                wh = tf.multiply(wh, self._rec_mask) / self._rec_keep_prob

            wh_h = tf.matmul(h, wh)

        with tf.name_scope("add"):
            logits = wx_x + wh_h + b

        with tf.name_scope("cell_transform"):
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=logits, num_or_size_splits=4, axis=1)

            new_c = (
                c * sigmoid(f) + sigmoid(i) * tf.tanh(j))

            new_h = tf.tanh(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)

        return new_h, new_state

    def gen_masks(self, session):
        return session.run({self._rec_mask: self._gen_rec_mask})  # , self._assign_mem_mask

    def update_drop_params(self, session, rec_keep_prob):
        session.run(self._rec_keep_prob_update, feed_dict={self._new_rec_keep_prob: rec_keep_prob})
