from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import rnn_cell_additions as dr
import optimizers
import logging

logger = logging.getLogger("logger")


class PTBModel(object):
    """class for handling the ptb model"""

    def __init__(self,
                 data,
                 config,
                 is_training):
        """the constructor builds the tensorflow_impl graph"""
        self._config = config
        self._is_training = is_training
        self._seed = config.seed
        self._gpu_devices = config.gpu_devices
        self._cpu_device = "/cpu:" + config.cpu_device
        self._debug_ops = list()
        self._stat_ops = list()

        self._activations = list()
        self._data = data

        self._init_scale = config.init_scale
        self._batch_size = config.batch_size
        self._layers = config.layers
        self._hid_size = config.hid_size
        self._bptt = config.bptt
        self._vocab_size = config.vocab_size
        self._embedding_size = config.embedding_size
        self._drop_e = config.drop_e
        self._drop_i = config.drop_i
        self._drop_h = config.drop_h
        self._drop_s = config.drop_s

        self._mos = config.mos
        if self._mos:
            self._mos_drop = config.mos_drop
            self._mos_experts_num = config.mos_experts_num

        with tf.name_scope("aux_variables"):
            with tf.name_scope("global_step"):
                self._global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope("epoch_counter"):
                self._epoch_count = tf.Variable(0, name='epoch', trainable=False)
                self._epoch_inc = tf.assign(self._epoch_count, tf.add(self._epoch_count, tf.constant(1)))
                self._epoch_reset = tf.assign(self._epoch_count, tf.constant(0))

        # construct the embedding layer on cpu device
        self._activations.append(self._data.input_data)

        self._build_embedding()

        self._build_rnn()

        self._build_loss()

        if self._is_training:
            # set learning rate as variable in order to anneal it throughout training
            with tf.name_scope("learning_rate"):
                self._lr = tf.Variable(config.lr, trainable=False, dtype=tf.float32)
                # a placeholder to assign a new learning rate
                self._new_lr = tf.placeholder(
                    tf.float32, shape=[], name="new_learning_rate")

                # function to update learning rate
                self._lr_update = tf.assign(self._lr, self._new_lr)

            # get trainable vars
            tvars = tf.trainable_variables()

            # define an optimizer with the averaged gradients
            with tf.name_scope("optimizer"):
                self._optimizer = []
                if config.opt == "sgd":
                    logger.debug("using SGD optimizer")
                    self._optimizer = optimizers.SGD(self, self._grads, tvars, self._config)
                    self._train_op = self._optimizer.train_op
                elif config.opt == "asgd":
                    logger.debug("using ASGD optimizer")
                    self._optimizer = optimizers.ASGD(self, self._grads, tvars, self._config)
                    self._train_op = self._optimizer.train_op
                # elif config.opt == "masgd":
                #     logger.info("using MASGD optimizer")
                #     opt = optimizers.ASGD(self, self._grads, tvars)
                #     self._optimizer = optimizers.MASGD(self, opt.updates, tvars)
                #     self._train_op = self._optimizer.train_op
                # elif config.opt == "rms":
                #     logger.info("using RMS optimizer")
                #     self._optimizer = optimizers.RMSprop(self, self._grads, tvars)
                #     self._train_op = self._optimizer.train_op
                # elif config.opt == "arms":
                #     logger.info("using ARMS optimizer")
                #     opt = optimizers.RMSprop(self, grads, tvars, use_opt=False)
                #     self._optimizer = optimizers.ASGD(self, opt.updates, tvars)
                #     self._train_op = self._optimizer.train_op
                # elif config.opt == "marms":
                #     logger.info("using MARMS optimizer")
                #     opt = optimizers.RMSprop(self, grads, tvars, use_opt=False)
                #     self._optimizer = optimizers.ASGD(self, opt.updates, tvars)
                #     self._train_op = self._optimizer.train_op
                else:
                    raise ValueError( config.opt + " is not a valid optimizer")

    def _build_embedding(self):
        init_scale = self._config.embed_init_scale
        init = tf.random_uniform(shape=[self._vocab_size, self._embedding_size],
                                 minval=-init_scale,
                                 maxval=init_scale,
                                 seed=self._seed,
                                 dtype=tf.float32)

        with tf.variable_scope("embedding"), tf.device(self._cpu_device):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            logger.debug("adding embedding matrix with dims [{:d}, {:d}]".format(self._vocab_size, self._embedding_size))
            self._embedding_map = tf.get_variable(name="embedding", dtype=tf.float32, initializer=init)
            embedding_vec = tf.nn.embedding_lookup(self._embedding_map, self._activations[-1])

            if self._is_training and (self._drop_e > 0 or self._drop_i > 0):
                with tf.name_scope("embedding_mask"):
                    # non variational wrapper for the embedding
                    logger.debug("adding embedding mask with dims [{:d}, {:d}, {:d}]"
                                 .format(self._batch_size, self._bptt, self._embedding_size))
                    self._emb_mask = tf.placeholder(dtype=tf.float32,
                                                    shape=[self._batch_size, self._bptt, self._embedding_size],
                                                    name="embedding_mask")
                    if self._drop_e > 0:
                        if self._config.drop_embed_var:
                            logger.debug("using variational embedding dropout")
                            random_tensor = ops.convert_to_tensor(1-self._drop_e)
                            random_tensor += \
                                random_ops.random_uniform([self._batch_size, 1, self._embedding_size],
                                                          seed=self._seed)
                            random_tensor = tf.tile(random_tensor, [1, self._bptt, 1])
                            self._gen_emb_mask = math_ops.floor(random_tensor)
                        else:
                            logger.debug("using naive embedding dropout")
                            random_tensor = ops.convert_to_tensor(1-self._drop_e)
                            random_tensor += \
                                random_ops.random_uniform([self._batch_size, self._bptt, self._embedding_size],
                                                          seed=self._seed)
                            self._gen_emb_mask = math_ops.floor(random_tensor)
                    else:
                        self._gen_emb_mask = tf.ones([self._batch_size, self._bptt, self._embedding_size])

                    embedding_vec = math_ops.div(embedding_vec, (1-self._drop_e)*(1-self._drop_i)) * self._emb_mask
        self._activations.append(embedding_vec)

    def _build_rnn(self):
        self._cell = list()
        self._initial_state = list()
        self._state = list()

        # define the lstm cell
        lstm_cell = self._build_lstm_cell()

        outputs = tf.unstack(self._activations[-1], num=self._bptt, axis=1)

        self._final_state = list()
        for i in range(self._layers):
            with tf.variable_scope("layer_%d" % (i+1)):
                self._cell.append(lstm_cell(self._hid_size[i]))
                self._initial_state.append(self._cell[-1].zero_state(self._batch_size, dtype=tf.float32))

                outputs, state = tf.nn.static_rnn(self._cell[-1], outputs, initial_state=self._initial_state[-1])

                self._final_state.append(state)
                output = tf.reshape(tf.concat(outputs, 1), [-1, self._hid_size[i]])

                self._activations.append(output)

    def _build_lstm_cell(self):
        def cell(lstm_size):
            if self._config.DC:
                logger.debug("using weight-dropped LSTM cell")
                return dr.WeightDroppedLSTMCell(num_units=lstm_size,
                                                is_training=self._is_training,
                                                state_is_tuple=True)
            else:
                logger.debug("using LSTM cell")
                return tf.contrib.rnn.LSTMBlockCell(num_units=lstm_size)

        final_cell = cell
        # if dropout is needed add a dropout wrapper
        if self._is_training and (self._drop_h[0] > 0 or self._drop_h[1] > 0 or
                                  self._drop_s[0] > 0 or self._drop_s[1] > 0):
            def final_cell(lstm_size):
                if self._config.variational is not None:
                    if self._config.DC:
                        logger.debug("using weight-dropped variational dropout")
                        return dr.WeightDroppedVariationalDropoutWrapper(cell(lstm_size),
                                                                         self._batch_size,
                                                                         lstm_size)
                    else:
                        logger.debug("using variational dropout")
                        return dr.VariationalDropoutWrapper(cell(lstm_size),
                                                            self._batch_size,
                                                            lstm_size)
                else:
                    raise ValueError("non variational dropout is deprecated")
        return final_cell

    def _get_prev_h(self, outputs):
        _, initial_h = self._initial_state[-1]
        state = list()
        state.append(initial_h)
        state.extend(outputs[:-1])
        state = tf.stack(state, axis=1)
        state = tf.reshape(state, [-1, self._hid_size[-1]])
        return state

    def _build_loss(self):
        if self._embedding_size == self._hid_size[-1] or self._mos:
            # outer softmax matrix is tied with embedding matrix
            logger.debug("tied embedding")
            w_out = tf.transpose(self._embedding_map)
        else:
            logger.debug("untied embedding")
            w_out = tf.get_variable(name="w_embed_out", shape=[self._hid_size[-1], self._vocab_size], dtype=tf.float32)

        b_out = tf.get_variable(name="b_out",
                                dtype=tf.float32,initializer=tf.zeros([self._vocab_size], dtype=tf.float32))

        with tf.name_scope("loss"):
            with tf.name_scope("data_loss"):
                if self._mos:
                    logger.debug("adding mos with %d contexts" % self._mos_experts_num)
                    logits = self._build_mos(w_out, b_out)
                else:
                    logger.debug("adding softmax layer")
                    logits = tf.matmul(self._activations[-1], w_out) +  b_out

            if self._is_training:
                random_tensor = ops.convert_to_tensor(1-self._config.drop_label)
                random_tensor += random_ops.random_uniform([self._batch_size * self._bptt], seed=self._seed)
                mask =  math_ops.floor(random_tensor)
            else:
                mask = tf.ones([self._batch_size * self._bptt], dtype=tf.float32)

            losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                        [tf.reshape(self._data.targets, [-1])],
                                                                        [mask])
            loss = tf.reduce_mean(losses)

            self._loss = loss

            if self._config.AR and self._is_training:
                logger.debug("using activation regularization")
                with tf.name_scope("AR"):
                    loss += self._config.AR * tf.reduce_mean(tf.square(tf.reshape(self._activations[-1], [-1, 1])))

            if self._config.TAR and self._is_training:
                logger.debug("using temporal activation regularization")
                with tf.name_scope("TAR"):
                    outputs_reshaped = tf.reshape(self._activations[-1], [self._batch_size, self._bptt, -1])
                    diff = outputs_reshaped[:, :-1, :] - outputs_reshaped[:, 1:, :]
                    loss += self._config.TAR * tf.reduce_mean(tf.square(tf.reshape(diff, [-1, 1])))

            if self._config.wdecay and self._is_training:
                logger.debug("using L2 regularization")
                for tvar in tf.trainable_variables():
                    loss += self._config.wdecay * tf.reduce_sum(tf.square(tf.reshape(tvar, [-1, 1])))


        with tf.name_scope("compute_grads"):
            self._grads = None
            if self._is_training:
                self._grads = tf.gradients(loss, tf.trainable_variables())

    def _build_mos(self, w_out, b_out):
        with tf.name_scope("mos"):
            # pi
            prior = tf.get_variable(name="mos_pi",
                                    shape=[self._hid_size[-1], self._mos_experts_num],
                                    dtype=tf.float32)
            # context vectors
            w_h = tf.get_variable(name="mos_w_h",
                                  shape=[self._hid_size[-1], self._mos_experts_num * self._embedding_size],
                                  dtype=tf.float32)
            b_h = tf.get_variable(name="mos_b_h",
                                  shape=[self._mos_experts_num * self._embedding_size],
                                  dtype=tf.float32)

            prior = tf.matmul(self._activations[-1], prior)
            pi = tf.nn.softmax(prior, name="mos_prior")

            h = tf.reshape(tf.tanh(tf.matmul(self._activations[-1], w_h) + b_h), [-1, self._embedding_size])

            if self._is_training:
                self._mos_mask = tf.placeholder(dtype=tf.float32,
                                                shape=[self._batch_size * self._bptt * self._mos_experts_num,
                                                       self._embedding_size],
                                                name="mos_mask")

                if self._config.variational is not None:
                    with tf.name_scope("mos_mask_gen"):
                        random_tensor = ops.convert_to_tensor(1-self._mos_drop)
                        random_tensor += random_ops.random_uniform(
                            [self._batch_size, 1, self._mos_experts_num * self._embedding_size], seed=self._seed)
                        random_tensor = tf.tile(random_tensor, [1, self._bptt, 1])
                        self._gen_mos_mask = tf.reshape(math_ops.floor(random_tensor),
                                                        [self._batch_size * self._bptt * self._mos_experts_num,
                                                         self._embedding_size])
                else:
                    with tf.name_scope("mos_mask_gen"):
                        random_tensor = ops.convert_to_tensor(1-self._mos_drop)
                        random_tensor += random_ops.random_uniform(
                            [self._batch_size * self._mos_experts_num * self._bptt, self._embedding_size],
                            seed=self._seed)
                        self._gen_mos_mask = math_ops.floor(random_tensor)

                h = math_ops.div(h, 1-self._mos_drop) * self._mos_mask

            a = tf.matmul(h, w_out) + b_out
            # mos
            a_mos = tf.reshape(tf.nn.softmax(a), [-1, self._mos_experts_num, self._vocab_size])
            pi = tf.reshape(pi, [-1, self._mos_experts_num, 1])
            weighted_softmax = tf.multiply(a_mos, pi)
            prob = tf.reduce_sum(weighted_softmax, axis=1)
            log_prob = tf.log(prob+1e-8)

        return log_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def bptt(self):
        return self._bptt

    @property
    def hid_size(self):
        return self._hid_size

    @property
    def init_scale(self):
        return self._init_scale

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def data(self):
        return self._data

    @property
    def lr(self):
        return self._lr

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch_count

    @property
    def config(self):
        return self._config

    @property
    def emb_mask(self):
        return self._emb_mask

    @property
    def stat_ops(self):
        return self._stat_ops

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def epoch_inc(self, session):
        return session.run(self._epoch_inc)

    def epoch_reset(self, session):
        return session.run(self._epoch_reset)

    def gen_masks(self, session):
        feed_dict = {}
        if (self._drop_h[0] > 0 or self._drop_h[1] > 0 or
                self._drop_s[0] > 0 or self._drop_s[1] > 0):
            for i in range(self._layers):
                feed_dict.update(self._cell[i].gen_masks(session))
        if self._config.mos:
            feed_dict.update({self._mos_mask: session.run(self._gen_mos_mask)})
        return feed_dict

    def gen_emb_mask(self, session):
        return {self._emb_mask: session.run(self._gen_emb_mask)}

    def gen_wdrop_mask(self, session):
        masks = {}
        if self._config.drop_s[0] > 0 or self._config.drop_s[1] > 0:
            for cell in self._cell:
                masks.update(cell.cell.gen_masks(session))
        return masks

    def update_drop_params(self, session, output_drop, state_drop):
        if (self._drop_h[0] > 0 or self._drop_h[1] > 0 or
                self._drop_s[0] > 0 or self._drop_s[1] > 0):
            for i in range(self._layers):
                if i < self._layers-1:
                    logger.info("layer %d: out %.2f, state %.2f" % (i+1, output_drop[0], state_drop[0]))
                    self._cell[i].update_drop_params(session,
                                                     1 - output_drop[0],
                                                     1 - state_drop[0])
                else:
                    logger.info("layer %d: out %.2f, state %.2f" % (i + 1, output_drop[1], state_drop[1]))
                    self._cell[i].update_drop_params(session,
                                                     1 - output_drop[1],
                                                     1 - state_drop[1])


