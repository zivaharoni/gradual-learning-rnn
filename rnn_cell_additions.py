import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

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
            self._output_mask = tf.Variable(tf.ones(shape=[self._batch_size, hidden_size]), trainable=False)
            self._state_mask = tf.Variable(tf.ones(shape=[self._batch_size, hidden_size]), trainable=False)
            self._mem_mask = tf.Variable(tf.ones(shape=[self._batch_size, hidden_size]), trainable=False)

            with tf.name_scope("out_mask"):

                random_tensor = ops.convert_to_tensor(self._output_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size])
                output_mask = math_ops.floor(random_tensor)
                self._assign_output_mask = tf.assign(self._output_mask, output_mask)

            with tf.name_scope("rec_mask"):
                random_tensor = ops.convert_to_tensor(self._state_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size])
                state_mask = math_ops.floor(random_tensor)
                self._assign_state_mask = tf.assign(self._state_mask, state_mask)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def update_masks(self, session):
        session.run([self._assign_state_mask, self._assign_output_mask])  # , self._assign_mem_mask

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
