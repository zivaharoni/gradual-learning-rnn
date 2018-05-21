# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger("logger")


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path, data_name):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, data_name + ".train.txt")
  valid_path = os.path.join(data_path, data_name + ".valid.txt")
  test_path = os.path.join(data_path, data_name + ".test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data):
        self.raw_data = data
        self.batch_size = batch_size = config.batch_size
        self.bptt = bptt = config.bptt
        self.epoch_size = epoch_size =  (len(data)-1) // (batch_size * bptt)
        self.data_len = data_len = epoch_size * batch_size * bptt
        self.data = np.reshape(data[:data_len], newshape=[batch_size, bptt*epoch_size])
        self.label = np.reshape(data[1:data_len+1], newshape=[batch_size, bptt*epoch_size])
        self.start_idx = 1
        self.input_data = tf.placeholder(dtype=tf.int32,shape=[batch_size, bptt], name="input")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, bptt], name="targets")

    def shuffle(self):
        data = self.raw_data
        data_len = self.data_len
        batch_size = self.batch_size
        epoch_size =  self.epoch_size
        bptt = self.bptt
        self.start_idx = start_idx = np.random.randint(0, (len(data)-1) % (self.batch_size * self.bptt))
        self.data = np.reshape(data[start_idx:start_idx + data_len], newshape=[batch_size, bptt * epoch_size])
        self.label = np.reshape(data[1+start_idx:data_len + start_idx + 1], newshape=[batch_size, bptt * epoch_size])

        logger.info("Batching from index %d" % self.start_idx)


    def get_batch(self, idx):
        return {self.input_data: self.data[:, idx:idx+self.bptt],
                self.targets: self.label[:, idx:idx + self.bptt]}

