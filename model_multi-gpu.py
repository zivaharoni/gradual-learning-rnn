from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import ptb_reader
import ptb_config
import os
import sys
import ast
import rnn_cell_additions as dr

flags = tf.flags
logging = tf.logging

# flags for distributed tf
flags.DEFINE_string(
    "gpu_devices", "0", "comma separated list of GPU numbers visible for the run. Example: '2,0,3' will "
                        "define gpu no.2 as gpu:0, gpu no.0 as gpu:1 and so on")
flags.DEFINE_string(
    "cpu_device", "0", "comma separated list of CPU numbers visible for the run.")

# flags for error handling
flags.DEFINE_string(
    "ckpt_file", None, "name of checkpoint file to load model from")
flags.DEFINE_integer(
    "start_layer", 0, "if restore is needed, mention the layer to start from")

# flags for model
flags.DEFINE_string(
    "name", "AWD-Try", "model name. describe shortly the purpose of the run")
flags.DEFINE_string(
    "model", "LWGC", "model type. options are:  Test, LWGC, GL, LAD, GL_LAD, Deep_GL_LAD")
flags.DEFINE_string(
    "data", "ptb", "dataset. options are: ptb, enwik8")
flags.DEFINE_integer(
    "seed", 0, "random seed used for initialization")
flags.DEFINE_integer(
    "batch_size", None, "#of sequences for all gpus in total")
flags.DEFINE_integer(
    "lstm_layers_num", None, "#of lstm layers")
flags.DEFINE_integer(
    "layer_epoch", None, "#of epochs per layer")
flags.DEFINE_string(
    "lwgc_grad_norm", None, "comma separated list of layers max grad norms. "
                                 "according to: embeddings, layer0, layer1, ...")
flags.DEFINE_float(
    "lr", None, "initial lr")
flags.DEFINE_float(
    "lr_decay", None, "lr decay when validation decreases")
flags.DEFINE_integer(
    "time_steps", None, "bptt truncation")
flags.DEFINE_bool(
    "GL", None, "gradual learning of the network")
flags.DEFINE_bool(
    "DC", False, "drop connect lstm's hidden-to-hidden connections")
flags.DEFINE_float(
    "AR", None, "activation regularization coefficient")
flags.DEFINE_float(
    "TAR", None, "temporal activation regularization coefficient")
flags.DEFINE_string(
    "opt", None, "sgd or asgd optimizer")
flags.DEFINE_integer(
    "embedding_size", None, "#of units in the embedding representation")
flags.DEFINE_integer(
    "units_num", None, "#of units in lstm cell")
flags.DEFINE_float(
    "keep_prob_embed", None, "keep prob of embedding representation unit")
flags.DEFINE_string(
    "drop_output", None, "keep prob of lstm output")
flags.DEFINE_string(
    "drop_state", None, "keep prob of lstm state")
FLAGS = flags.FLAGS

gpu_list = FLAGS.gpu_devices.split(",")


def get_device_list(device, nums):
    device_list = []
    visible_device_list = []
    num_devices = 0
    nums_list = nums.split(",")
    for n, val in enumerate(nums_list):
        device_list.append('/' + device + ':' + str(n))
        visible_device_list.append('/' + device + ':' + val)
        num_devices += 1
    return device_list, visible_device_list, num_devices


class PTBModel(object):
    """class for handling the ptb model"""

    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """the constructor builds the tensorflow graph"""
        self._input = inputs
        vocab_size = config.vocab_size  # num of possible words
        self._gpu_devices, self._visible_gpu_devices, self._gpu_num = get_device_list("gpu", FLAGS.gpu_devices)
        self._cpu_device, _, _ = get_device_list("cpu", FLAGS.cpu_device)
        self._cpu_device = self._cpu_device[0] #ensure work on a single CPU
        with tf.name_scope("model_variables"):
            with tf.name_scope("global_step"):
                self._global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope("epoch_counter"):
                self._epoch_count = tf.Variable(0, name='epoch', trainable=False)
                self._epoch_inc = tf.assign(self._epoch_count, tf.add(self._epoch_count, tf.constant(1)))
                self._epoch_reset = tf.assign(self._epoch_count, tf.constant(0))

        self._cell = []
        self._initial_state = []
        self._final_state = []

        # construct the embedding layer on cpu:0
        with tf.variable_scope("embedding"), tf.device(self._cpu_device):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            embedding_map = tf.get_variable(
                name="embedding", shape=[vocab_size, config.embedding_size], dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-config.emb_init_scale, config.emb_init_scale))
            b_embed_in = tf.get_variable(name="b_embed_in", shape=[config.embedding_size], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-config.emb_init_scale, config.emb_init_scale))
            embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            if is_training and config.keep_prob_embed < 1:
                embedding_out = tf.nn.dropout(embedding,
                                              config.keep_prob_embed)
            else:
                embedding_out = embedding

        # split input to devices if needed
        with tf.name_scope("split_inputs"):
            if self._gpu_num > 1:
                embedding_out = tf.split(embedding_out, self._gpu_num)
                targets = tf.split(inputs.targets, self._gpu_num)
            else:
                embedding_out = [embedding_out]
                targets = [inputs.targets]

        # construct the rest of the model on every gpu
        all_loss = []  # 2D array [i,j] element stands for the loss of the j-th layer of the i-th gpu
        all_grads = []  # 2D array [i,j] element stands for the grad of the j-th layer of the i-th gpu

        with tf.variable_scope("gpus"):
            for i in range(self._gpu_num):
                with tf.device(self._gpu_devices[i]), tf.name_scope("gpu-%d" % i):
                    loss, grads, cell, initial_state, final_state = self.complete_model(embedding_out[i],
                                                                                        embedding_map,
                                                                                        config,
                                                                                        is_training,
                                                                                        inputs,
                                                                                        targets[i])

                    self._cell.append(cell)
                    self._initial_state.append(initial_state)
                    self._final_state.append(final_state)
                    all_loss.append(loss)
                    all_grads.append(grads)

                    # reuse variables for the next gpu
                    tf.get_variable_scope().reuse_variables()

        # reduce per-gpu-loss to total loss
        with tf.name_scope("reduce_loss"):
            self._loss = self.reduce_loss(all_loss)

        if is_training:
            # average grads ; sync point
            grads = []
            with tf.name_scope("average_grads"):
                averaged_grads = self.average_grads(all_grads)
            if config.LWGC:
                if config.GL:
                    for i in range(config.lstm_layers_num):
                        grad = averaged_grads[i]
                        grad_clipped_emb, _ = tf.clip_by_global_norm([grad[0], grad[1], grad[-1]], config.max_grad_norm[0])
                        grad_vec = [grad_clipped_emb[0], grad_clipped_emb[1]]
                        grad_res = [grad_clipped_emb[-1]]
                        for j in range(config.lstm_layers_num):
                            grads_layer = [grad[3 * j + 2], grad[3 * j + 3], grad[3 * j + 4]] if config.DC \
                                else [grad[2 * (j + 1)], grad[2 * (j + 1) + 1]]
                            grad_clipped, _ = tf.clip_by_global_norm(grads_layer, config.max_grad_norm[j + 1])
                            grad_vec += grad_clipped
                        grad_vec += grad_res
                        grads.append(grad_vec)
                else:
                    grad = averaged_grads[-1]
                    grad_clipped_emb, _ = tf.clip_by_global_norm([grad[1], grad[-1]], config.max_grad_norm[0])
                    grad_vec = [grad[0], grad_clipped_emb[0]]
                    grad_res = [grad_clipped_emb[-1]]
                    for j in range(config.lstm_layers_num):
                        grads_layer = [grad[3*j + 2], grad[3*j + 3], grad[3*j + 4]] if config.DC \
                            else [grad[2 * (j + 1)], grad[2 * (j + 1) + 1]]
                        grad_clipped, _ = tf.clip_by_global_norm(grads_layer, config.max_grad_norm[j + 1])
                        grad_vec += grad_clipped
                    grad_vec += grad_res
                    grads.append(grad_vec)
            else:
                grad_clipped, _ = tf.clip_by_global_norm(averaged_grads[-1], config.max_grad_norm[0])
                grads.append(grad_clipped)
            # set learning rate as variable in order to anneal it throughout training
            with tf.name_scope("learning_rate"):
                self._lr = tf.Variable(0.0, trainable=False)
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
                if config.GL is True:
                    for i in range(config.lstm_layers_num):
                        if config.opt == "sgd":
                            self._optimizer.append(SGDOptimizer(self, grads[i], tvars))
                        elif config.opt == "asgd":
                            self._optimizer.append(ASGDOptimizer(self, grads[i], tvars))
                        else:
                            raise ValueError("must choose a valid optimizer")
                else:
                    if config.opt == "sgd":
                        self._optimizer.append(SGDOptimizer(self, grads[-1], tvars))
                    elif config.opt == "asgd":
                        self._optimizer.append(ASGDOptimizer(self, grads[-1], tvars))
                    else:
                        raise ValueError("must choose a valid optimizer")

    def average_grads(self, all_grads):
        """ average the grads of the currently trained layer

        Args:
            grads: 2D array, the [i,j] element stands for the loss of the j-th layer of the i-th gpu

        Returns:
            grads: a list of the averaged grads for each layer
        """

        if self._gpu_num == 1:
            average_layer_grads = all_grads[0]
        else:
            average_layer_grads = []

            if config.GL is True:
                for layer in range(config.lstm_layers_num):
                    layer_grads = [all_grads[i][layer][:] for i in range(len(all_grads))]
                    grads = []
                    for grad_and_vars in zip(*layer_grads):
                        # Note that each grad_and_vars looks like the following:
                        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                        gpu_grads = []

                        for g in grad_and_vars:
                            if g is not None:
                                # Add 0 dimension to the gradients to represent the tower.
                                expanded_g = tf.expand_dims(g, 0)

                                # Append on a 'tower' dimension which we will average over below.
                                gpu_grads.append(expanded_g)
                        if g is not None:
                            # Average over the 'tower' dimension.
                            grad = tf.concat(axis=0, values=gpu_grads)
                            grad = tf.reduce_mean(grad, 0)
                        else:
                            grad = g

                        grads.append(grad)
                    average_layer_grads.append(grads)
            else:
                layer_grads = [all_grads[i][-1][:] for i in range(len(all_grads))]
                grads = []
                for grad_and_vars in zip(*layer_grads):
                    # Note that each grad_and_vars looks like the following:
                    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                    gpu_grads = []

                    for g in grad_and_vars:
                        if g is not None:
                            # Add 0 dimension to the gradients to represent the tower.
                            expanded_g = tf.expand_dims(g, 0)

                            # Append on a 'tower' dimension which we will average over below.
                            gpu_grads.append(expanded_g)
                    if g is not None:
                        # Average over the 'tower' dimension.
                        grad = tf.concat(axis=0, values=gpu_grads)
                        grad = tf.reduce_mean(grad, 0)
                    else:
                        grad = g

                    grads.append(grad)
                average_layer_grads.append(grads)
        return average_layer_grads

    def reduce_loss(self, all_loss):
        """ average the loss obtained by gpus

        Args:
            all_loss: 2D array, the [i,j] element stands for the loss of the j-th layer of the i-th gpu

        Returns:
            grads: a list of the loss for each layer
        """
        if self._gpu_num == 1:
            total_loss = all_loss[0]
        else:
            total_loss = []
            if config.GL is True:
                for i in range(config.lstm_layers_num):
                    if self._gpu_num > 1:
                        layer_loss = [all_loss[j][i] for j in range(self._gpu_num)]
                        total_loss.append(tf.reduce_mean(layer_loss))
                    else:
                        total_loss.append(all_loss[0][i])
            else:
                total_loss.append(all_loss[0][-1])
        return total_loss

    def complete_model(self, embedding_out, embedding_map, config, is_training, inputs, targets):
        """ Build rest of model for a single gpu

        Args:
            embedding_out: the embedding representation to be processed

        Returns:
            loss: a list for the loss calculated for each layer.
            grads: a list for the grads calculated for each loss.
        """

        batch_size = inputs.batch_size // self._gpu_num  # num of sequences
        assert inputs.batch_size // self._gpu_num == inputs.batch_size / self._gpu_num, \
            "must choose batch size that is divided by gpu_num"

        time_steps = config.time_steps  # num of time steps used in BPTT
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num  # num of units in the hidden layer

        # define basic lstm cell
        def lstm_cell(lstm_size):
            if config.DC:
                return dr.WeightDroppedLSTMCell(num_units=lstm_size,
                                                is_training=is_training,
                                                state_is_tuple=True)
            else:
                return tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size,
                                                    forget_bias=config.forget_bias_init,
                                                    state_is_tuple=True)

        possible_cell = lstm_cell
        # if dropout is needed add a dropout wrapper
        if is_training and config.drop_output is not None:
            def possible_cell(lstm_size):
                if config.variational is not None:
                    if config.DC:
                        return dr.WeightDroppedVariationalDropoutWrapper(lstm_cell(lstm_size),
                                                                         batch_size,
                                                                         lstm_size)
                    else:
                        return dr.VariationalDropoutWrapper(lstm_cell(lstm_size),
                                                            batch_size,
                                                            lstm_size)
                else:
                    return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size),
                                                         output_keep_prob=config.drop_output)

        # organize layers' outputs and states in a list
        cell = []
        initial_state = []
        outputs = []
        state = []
        lstm_output = []
        for _ in range(config.lstm_layers_num):
            outputs.append([])
            state.append([])

        # unroll the cell to "time_steps" times
        # first lstm layer

        with tf.variable_scope("lstm%d" % 1):
            lstm_size = units_num
            cell.append(possible_cell(lstm_size))
            initial_state.append(cell[0].zero_state(batch_size, dtype=tf.float32))
            state[0] = initial_state[0]
            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, lstm_size]))

        # rest of layers
        for i in range(1, config.lstm_layers_num):
            with tf.variable_scope("lstm%d" % (i + 1)):
                lstm_size = config.embedding_size if i == (config.lstm_layers_num - 1) else units_num
                cell.append(possible_cell(lstm_size))
                initial_state.append(cell[i].zero_state(batch_size, dtype=tf.float32))
                state[i] = initial_state[i]
                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)
                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, lstm_size]))

        # outer embedding bias
        b_embed_out = tf.get_variable(name="b_embed_out", shape=[vocab_size], dtype=tf.float32)
        # outer softmax matrix is tied with embedding matrix
        w_out = tf.transpose(embedding_map)

        # get trainable vars
        tvars = tf.trainable_variables()

        # since using GL we have logits, losses and cost for every layer
        logits = []
        losses = []
        loss = []
        grads = []
        tvars_list = tf.trainable_variables()

        if config.GL is True:
            for i in range(config.lstm_layers_num):
                with tf.variable_scope("loss" + str(i + 1)):
                    logits.append(tf.matmul(lstm_output[i], w_out) + b_embed_out)
                    losses.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits[i]],
                                                                                     [tf.reshape(targets, [-1])],
                                                                                     [tf.ones([batch_size * time_steps],
                                                                                              dtype=tf.float32)]))
                    if config.AR and is_training:
                        with tf.name_scope("AR"):
                            for j in range(i+1):
                                losses[-1] += config.AR * tf.reduce_mean(tf.square(tf.reshape(lstm_output[j], [-1, 1])))

                    if config.TAR and is_training:
                        with tf.name_scope("TAR"):
                            for j in range(i+1):
                                outputs_reshaped = tf.reshape(lstm_output[j], [config.batch_size, config.time_steps, -1])
                                diff = outputs_reshaped[:, :-1, :] - outputs_reshaped[:, 1:, :]
                                losses[-1] += config.TAR * tf.reduce_mean(tf.square(tf.reshape(diff, [-1, 1])))
                    loss.append(tf.reduce_sum(losses[i]) / batch_size)
                    grad = tf.gradients(loss[i], tvars_list)
                    grads.append(grad)
        else:
            logits.append(tf.matmul(lstm_output[-1], w_out) + b_embed_out)
            losses.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits[-1]],
                                                                             [tf.reshape(targets, [-1])],
                                                                             [tf.ones([batch_size * time_steps],
                                                                                      dtype=tf.float32)]))
            loss.append(tf.reduce_sum(losses[-1]) / batch_size)
            grad = tf.gradients(loss[-1], tvars_list)
            grads.append(grad)

        final_state = state

        return loss, grads, cell, initial_state, final_state

    def initial_state(self, device_num):
        return self._initial_state[device_num]

    def final_state(self, device_num):
        return self._final_state[device_num]

    def loss(self, layer=-1):
        return self._loss[layer]

    def train_op(self, layer=-1):
        return self._optimizer[layer].train_op

    def optimizer(self, layer=-1):
        return self._optimizer[layer]

    @property
    def input(self):
        return self._input

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
    def gpu_num(self):
        return self._gpu_num

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def epoch_inc(self, session):
        return session.run(self._epoch_inc)

    def epoch_reset(self, session):
        return session.run(self._epoch_reset)

    def gen_masks(self, session):
        feed_dict = {}
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
                feed_dict.update(self._cell[j][i].gen_masks(session))
        return feed_dict

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
                print("layer %d: out %.2f, state %.2f" % (i+1, output_keep_prob[i], state_keep_prob[i]))
                self._cell[j][i].update_drop_params(session,
                                                    output_keep_prob[i],
                                                    state_keep_prob[i])

class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.time_steps = time_steps = config.time_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // time_steps
        self.input_data, self.targets = ptb_reader.ptb_producer(
            data, batch_size, time_steps, name=name)


class SGDOptimizer(object):
    def __init__(self, model, grads, tvars):

        optimizer = tf.train.GradientDescentOptimizer(model.lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=model.global_step)

    @property
    def train_op(self):
        return self._train_op


class ASGDOptimizer(object):

    count = 0

    def __init__(self, model, grads, tvars):
        optimizer = tf.train.GradientDescentOptimizer(model.lr)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)

        self._trigger = tf.get_variable("ASGD_trigger%d" % type(self).count, initializer=tf.constant(False, dtype=tf.bool), trainable=False)
        self._set_trigger = tf.assign(self._trigger, True)

        self._T = tf.get_variable("T%d" % type(self).count, initializer=tf.constant(0, dtype=tf.int32), trainable=False)
        self._new_T = tf.placeholder(tf.int32, shape=[], name="new_T%d" % type(self).count)
        self._set_T = tf.assign(self._T, self._new_T)

        self._avg_vars = []
        self._avg_assign_op = []
        self._save_tmp_vars = []
        self._load_tmp_vars = []
        for var in tvars:
            self._avg_vars.append(tf.get_variable(var.op.name + "final%d" % type(self).count,
                                                  initializer=tf.zeros_like(var, dtype=tf.float32),
                                                  trainable=False))
            with tf.name_scope("final_average"):
                self._avg_assign_op.append(tf.assign(var, self._avg_vars[-1] /
                                                     tf.cast((config.entire_network_epoch - self._T + 1) *
                                                     model.input.epoch_size, dtype=tf.float32)))
            with tf.name_scope("tmp_vars_holder"):
                tmp_vars_holder = tf.get_variable(var.op.name + "tmp%d" % type(self).count,
                                                  initializer=tf.zeros_like(var, dtype=tf.float32),
                                                  trainable=False)
                self._save_tmp_vars.append(tf.assign(tmp_vars_holder, var))
                self._load_tmp_vars.append(tf.assign(var, tmp_vars_holder))


        def trigger_on():
            with tf.name_scope("trigger_is_on"):
                op = list()
                op.append(tf.identity(self._trigger))
                op.append(tf.identity(self._T))
                for i, var in enumerate(tvars):
                    op.append(tf.assign_add(self._avg_vars[i], var))

            return op

        def trigger_off():
            with tf.name_scope("trigger_is_off"):
                op = list()
                op.append(tf.identity(self._trigger))
                op.append(tf.identity(self._T))
                for i, var in enumerate(tvars):
                    op.append(tf.identity(self._avg_vars[i]))

            return op

        with tf.name_scope("trigger_mux"):
            self._update_op = tf.cond(self._trigger, lambda: trigger_on(), lambda: trigger_off())

        type(self).count += 1

    def set_trigger(self, session):
        return session.run(self._set_trigger)

    @property
    def train_op(self):
        return self._train_op

    @property
    def trigger(self):
        return self._trigger

    @property
    def update_op(self):
        return self._update_op

    def set_T(self, session, T):
        return session.run(self._set_T, feed_dict={self._new_T: T})

    @property
    def avg_assign_op(self):
        return self._avg_assign_op

    @property
    def save_tmp_vars(self):
        return self._save_tmp_vars

    @property
    def load_tmp_vars(self):
        return self._load_tmp_vars


def print_tvars():
    tvars = tf.trainable_variables()
    nvars = 0
    for var in tvars[1:]:
        sh = var.get_shape().as_list()
        nvars += np.prod(sh)
        # print ('var: %s, size: [%s]' % (var.name,', '.join(map(str, sh))))
    print (nvars, ' total variables')


def run_epoch(session, model, eval_op=None, verbose=True, layer=-1):
    """run the given model over its data"""

    start_time = time.time()
    losses = 0.0
    iters = 0

    # zeros initial state for all devices
    state = []
    for k in range(model.gpu_num):
        state.append(session.run(model.initial_state(k)))

    feed_dict_masks = {}
    # if variational every epoch --> update masks
    if config.variational == 'epoch' and eval_op is not None:
        feed_dict_masks = model.gen_masks(session)

    # evaluate loss and final state for all devices
    fetches = {
        "loss": model.loss(layer),
    }
    for k in range(model.gpu_num):
        fetches["final_state%d" % k] = model.final_state(k)

    # perform train op if training
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        # if variational every batch --> update masks
        if config.variational == 'batch' and eval_op is not None:
            feed_dict_masks = model.gen_masks(session)

        # pass states between time batches
        feed_dict = dict(feed_dict_masks.items())
        for i in range(model.gpu_num):
            gpu_state = model.initial_state(i)
            for j, (c, h) in enumerate(gpu_state):
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        # for key,val in feed_dict.items():
        #    print(key, val)

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        for k in range(model.gpu_num):
            state[k] = vals["final_state%d" % k]

        losses += loss
        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(losses / iters)


def asgd_run_epoch(session, model, eval_op=None, verbose=True, layer=-1):
    """run the given model over its data"""

    start_time = time.time()
    losses = 0.0
    iters = 0

    # zeros initial state for all devices
    state = []
    for k in range(model.gpu_num):
        state.append(session.run(model.initial_state(k)))

    feed_dict_masks = {}
    # if variational every epoch --> update masks
    if config.variational == 'epoch' and eval_op is not None:
        feed_dict_masks = model.gen_masks(session)

    # evaluate loss and final state for all devices
    fetches = {
        "loss": model.loss(layer),
    }
    for k in range(model.gpu_num):
        fetches["final_state%d" % k] = model.final_state(k)

    # perform train op if training
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        fetches["update_op"] = model.optimizer(layer).update_op

    for step in range(model.input.epoch_size):
        # if variational every batch --> update masks
        if config.variational == 'batch' and eval_op is not None:
            feed_dict_masks = model.gen_masks(session)

        # pass states between time batches
        feed_dict = dict(feed_dict_masks.items())
        for i in range(model.gpu_num):
            gpu_state = model.initial_state(i)
            for j, (c, h) in enumerate(gpu_state):
                feed_dict[c] = state[i][j].c
                feed_dict[h] = state[i][j].h

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        for k in range(model.gpu_num):
            state[k] = vals["final_state%d" % k]

        losses += loss
        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                   iters * model.input.batch_size / (time.time() - start_time)))

        if step == model.input.epoch_size - 1 and eval_op is not None:
            print("trigger: ", vals["update_op"][0], "\tT: ", vals["update_op"][1])

    return np.exp(losses / iters)


def train_optimizer(session, layer, m, mvalid, mtest, train_writer, valid_writer, test_writer, saver):
    """ Trains the network by the given optimizer """
    global bestVal
    epochs_num = config.layer_epoch if layer != -1 else config.entire_network_epoch

    run_e = run_epoch if config.opt == "sgd" else asgd_run_epoch
    m.update_drop_params(session, config.drop_output[layer], config.drop_state[layer])
    print("updating output keep probability to", config.drop_output[layer])
    print("updating state keep probability to", config.drop_state[layer])

    if FLAGS.ckpt_file is None:
        current_lr = config.lr
        m.assign_lr(session, current_lr)
    else:
        current_lr = session.run(m.lr)

    valid_perplexity = []
    valid_avg_perplexity = []
    i = session.run(m.epoch)
    if config.opt == "asgd":
        validation_tolerance = 5

    while i < epochs_num:
        start_time = time.time()
        if len(valid_perplexity) >= 2 and valid_perplexity[-1] > valid_perplexity[-2]:
            current_lr *= config.lr_decay
            m.assign_lr(session, current_lr)
            lr_sum = tf.Summary(value=[tf.Summary.Value(tag="learning_rate_track" + str(layer),
                                                       simple_value=current_lr)])
            train_writer.add_summary(lr_sum, i + 1)
            if config.opt == "asgd" and not session.run(m.optimizer(layer).trigger):
                validation_tolerance -= 1
                if validation_tolerance == 0:
                    print("setting trigger and T")
                    m.optimizer(layer).set_trigger(session)
                    m.optimizer(layer).set_T(session, i + 1)
                    config.lr_decay = 1.0

        print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_e(session, m, eval_op=m.train_op(layer), verbose=True, layer=layer)
        train_sum = tf.Summary(value=[tf.Summary.Value(tag="train_perplexity_layer" + str(layer),
                                                       simple_value=train_perplexity)])
        train_writer.add_summary(train_sum, i + 1)
        print("Epoch: %d Train Perplexity: %.3f Bits: %.3f " % (i + 1, train_perplexity, np.log2(train_perplexity)))

        if i == epochs_num - 1 and config.opt is "asgd":
            print("setting averaged weights....")
            session.run(m.optimizer(layer).avg_assign_op)

        valid_perplexity.append(run_e(session, mvalid, verbose=False, layer=layer))
        valid_sum = tf.Summary(value=[tf.Summary.Value(tag="valid_perplexity_layer" + str(layer),
                                                       simple_value=valid_perplexity[-1])])
        session.run(m.optimizer(layer).save_tmp_vars)  # saving the current vars
        session.run(m.optimizer(layer).avg_assign_op)  # assign avg vars for evaluation
        valid_avg_perplexity.append(run_e(session, mvalid, verbose=False, layer=layer))
        valid_avg_sum = tf.Summary(value=[tf.Summary.Value(tag="valid_perplexity_layer" + str(layer),
                                                       simple_value=valid_avg_perplexity[-1])])
        session.run(m.optimizer(layer).load_tmp_vars)  # loading back the current vars
        valid_writer.add_summary(valid_sum, i + 1)
        valid_writer.add_summary(valid_avg_sum, i + 1)
        print("Epoch: %d, Valid Perplexity: %.3f, Avg Valid Perplexity: %.3f, Bits: %.3f" % (i + 1,
                                                                                             valid_perplexity[-1],
                                                                                             valid_avg_perplexity[-1],
                                                                                             np.log2(valid_perplexity[-1])))

        elapsed = time.time() - start_time
        print("Epoch: %d took %02d:%02d" % (i + 1, elapsed // 60, elapsed % 60))

        i = m.epoch_inc(session)
        # save model only when validation improves
        if bestVal > valid_perplexity[-1]:
            bestVal = valid_perplexity[-1]
            try:
                save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
                print("save path is: %s" % save_path)
            except:
                pass

    m.epoch_reset(session)


def read_flags(config, FLAGS):
    # assign flags into config
    flags_dict = FLAGS.__dict__['__flags']
    for key, val in flags_dict.items():
        if val is not None:
            if key.startswith("drop"):
                val = ast.literal_eval(val)
            if key == "layer_epoch":
                setattr(config, "entire_network_epoch", val)

            setattr(config, key, val)
    # split max grad norm
    if config.LWGC:
        lwgc_grad_norm = FLAGS.lwgc_grad_norm if FLAGS.lwgc_grad_norm is not None else config.lwgc_grad_norm
        config.max_grad_norm = ast.literal_eval(lwgc_grad_norm)
    else:
        config.global_max_grad_norm = config.max_grad_norm[0]
    return config


def get_config(config_name):
    if config_name == "LAD":
        return config_pool.LADConfig()
    elif config_name == "Test":
        return config_pool.TestConfig()
    elif config_name == "LWGC":
        return config_pool.LWGCConfig()
    elif config_name == "DC":
        return config_pool.DropConnectConfig()
    elif config_name == "asgd":
        return config_pool.ASGDConfig()
    elif config_name == "reduced_emb":
        return config_pool.ReducedEmbeddingConfig()
    elif config_name == "AR":
        return config_pool.ARConfig()
    elif config_name == "TAR":
        return config_pool.TARConfig()
    elif config_name == "benchmark":
        return config_pool.BenchmarkConfig()
    elif config_name == "debug":
        return config_pool.DebugConfig()
    elif config_name == "LAD":
        return config_pool.LADConfig()
    else:
        raise ValueError("Invalid model: %s", model_config_name)


def print_config(config, sim_name):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    print('\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))
    file_name = '../results/' + sim_name + '/config.txt'
    config_file = open(file_name, "w")
    config_file.write('\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))
    config_file.close()


def get_simulation_name(config):
    name = config.name + '_model-' + config.model + '_emb' + str(config.embedding_size) + '-'
    name += str(config.lstm_layers_num) + 'x' + str(config.units_num)
    if config.GL:
        name += '_GL'
    if config.DC:
        name += '_DC'
    if config.AR:
        name += '_AR'
    if config.TAR:
        name += '_TAR'
    if config.LWGC:
        name += '_LWGC-'
        for i in range(config.lstm_layers_num+1):
            name += str(config.max_grad_norm[i]) + '-'
    name += '_seed-' + str(config.seed)
    return name

if FLAGS.data == 'ptb':
    config_pool = ptb_config
    reader = ptb_reader.ptb_raw_data
else:
    raise ValueError("Invalid data: %s", FLAGS.data)


config = get_config(FLAGS.model)
config = read_flags(config, FLAGS)

seed = FLAGS.seed

simulation_name = get_simulation_name(config)
model_config_name = FLAGS.model
directory = "../results/" + simulation_name
data_path = "../data"

print("-" * 89)
print("results are saved into:", directory)
print("-" * 89)

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + '/train')
    os.makedirs(directory + '/test')
    os.makedirs(directory + '/saver')

raw_data = reader(data_path)
train_data, valid_data, test_data, _ = raw_data

print("\n\n", " ".join(sys.argv), "\n")
print("-" * 89)
print_config(config, simulation_name)
print("-" * 89)

bestVal = config.vocab_size


def main(_):

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        tf.set_random_seed(seed)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, inputs=train_input)

            train_writer = tf.summary.FileWriter(directory + '/train',
                                                 graph=tf.get_default_graph())

            tf.summary.scalar("Learning Rate", m.lr)
            tf.summary.scalar("global step", m.global_step)

        with tf.name_scope("Valid"):
                valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)

                valid_writer = tf.summary.FileWriter(directory + '/valid')

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        tvars_res = tf.trainable_variables()
        if config.restore and config.lstm_layers_num > 1:
            rmv_idx = 2*config.lstm_layers_num
            del tvars_res[rmv_idx]
            del tvars_res[rmv_idx]
        trainable_saver = tf.train.Saver(var_list=tvars_res)

    gpu_devices_num_list = ""
    for g in range(m._gpu_num):
        gpu_devices_num_list += gpu_list[g][-1] + ','
    gpu_devices_num_list = gpu_devices_num_list[:-1]
    gpu_options = tf.GPUOptions(visible_device_list=gpu_devices_num_list)
    sess_config = tf.ConfigProto(device_count={"CPU": 2}, gpu_options=gpu_options)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        if FLAGS.start_layer is not None and FLAGS.start_layer > 0:
            session.run(tf.global_variables_initializer())
            saved_model_path = FLAGS.ckpt_file #load from ckpt file
            print("\nloading model from: " + saved_model_path)
            trainable_saver.restore(session, saved_model_path)
            start_layer = FLAGS.start_layer - 1
        else:
            session.run(tf.global_variables_initializer())
            start_layer = 0

        # Gradual Training
        if config.GL:
            print("\ntraining gradually...\n")
            for layer in range(start_layer, config.lstm_layers_num):
                print("training layer #%d" % (layer + 1))
                start_time = time.time()
                layer_seed = seed + layer
                tf.set_random_seed(layer_seed)
                train_optimizer(session, layer, m, mvalid, _, train_writer, valid_writer, _, saver)
                elapsed = time.time() - start_time
                print("optimization of layer %d took %02d:%02d:%02d\n" %
                      (layer + 1, elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
                # save model
                saver.save(session, directory + '/saver/model', global_step=layer+1)
        else:
            print("\ntraining the entire network...\n")

            start_time = time.time()
            train_optimizer(session, -1, m, mvalid, _, train_writer, valid_writer, _, saver)
            end_time = time.time()
            elapsed = end_time - start_time
            print("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

        coord.request_stop()
        coord.join(threads)

    train_writer.close()
    valid_writer.close()


# TODO: estimator (model without parallelization)
# TODO: add comments on code and function declarations


if __name__ == "__main__":
    tf.app.run()
