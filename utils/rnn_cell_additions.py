
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
import numbers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


core_rnn_cell = tf.contrib.rnn
# _checked_scope = core_rnn_cell_impl._checked_scope


class VariationalDropoutWrapper(core_rnn_cell.RNNCell):
    def __init__(self, cell, batch_size, hidden_size, output_keep_prob=1.0, state_keep_prob=1.0):
        self._cell = cell
        self._output_keep_prob = output_keep_prob
        self._state_keep_prob = state_keep_prob
        # self._mem_keep_prob = 1.0
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

            # with tf.name_scope("mem_mask"):
            #     random_tensor = ops.convert_to_tensor(self._mem_keep_prob)
            #     random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size])
            #     mem_mask = math_ops.floor(random_tensor)
            #     self._assign_mem_mask = tf.assign(self._mem_mask, mem_mask)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def update_masks(self, session):
        session.run([self._assign_state_mask, self._assign_output_mask])  # , self._assign_mem_mask

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
            new_c = c  # math_ops.div(c, self._mem_keep_prob) * self._mem_mask
            new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h_fb)

        return new_h_out, new_state


class MoonDropoutWrapper(core_rnn_cell.RNNCell):
    def __init__(self, cell, batch_size, hidden_size, mem_keep_prob=1.0):
        self._cell = cell
        self._mem_keep_prob = mem_keep_prob
        self._batch_size = batch_size
        with tf.name_scope("moon_masks"):
            self._mem_mask = tf.Variable(tf.ones(shape=[self._batch_size, hidden_size]), trainable=False)

            with tf.name_scope("mem_mask"):
                random_tensor = ops.convert_to_tensor(self._mem_keep_prob)
                random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size])
                mem_mask = math_ops.floor(random_tensor)
                self._assign_mem_mask = tf.assign(self._mem_mask, mem_mask)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def update_masks(self, session):
        session.run([self._assign_state_mask, self._assign_output_mask])  # , self._assign_mem_mask

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        output, state = self._cell(inputs, state, scope)
        (c, h) = state
        with tf.name_scope("Dropout"):
            if (isinstance(self._output_keep_prob, float) and
               self._output_keep_prob == 1):
                return
            new_c = math_ops.div(c, self._mem_keep_prob) * self._mem_mask
            new_state = core_rnn_cell.LSTMStateTuple(new_c, h)

        return h, new_state


class DropoutWrapper(core_rnn_cell.RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        """Create a cell with added input and/or output dropout.

        Dropout is never used on the state.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d"
                             % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, scope)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state


class VariationalDropoutWrapperModified(core_rnn_cell.RNNCell):
    def __init__(self, cell, batch_size, hidden_size, output_keep_prob=1.0, state_keep_prob=1.0):
        self._cell = cell
        self._new_output_keep_prob = tf.placeholder(tf.float32, shape=[], name="output_keep_prob")
        self._new_state_keep_prob = tf.placeholder(tf.float32, shape=[], name="state_keep_prob")
        self._output_keep_prob = tf.Variable(output_keep_prob, trainable=False)
        self._state_keep_prob = tf.Variable(state_keep_prob, trainable=False)

        self._output_keep_prob_update = tf.assign(self._output_keep_prob, self._new_output_keep_prob)
        self._state_keep_prob_update = tf.assign(self._state_keep_prob, self._new_state_keep_prob)

        # self._mem_keep_prob = 1.0
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

            # with tf.name_scope("mem_mask"):
            #     random_tensor = ops.convert_to_tensor(self._mem_keep_prob)
            #     random_tensor += random_ops.random_uniform([self._batch_size, self._cell.output_size])
            #     mem_mask = math_ops.floor(random_tensor)
            #     self._assign_mem_mask = tf.assign(self._mem_mask, mem_mask)

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
            new_c = c  # math_ops.div(c, self._mem_keep_prob) * self._mem_mask
            new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h_fb)

        return new_h_out, new_state
