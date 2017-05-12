from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import reader
import os
import rnn_cell_additions as dr


class PTBModel(object):
    """class for handling the learning of the ptb model"""

    # batch_size,time_steps,vocab_size,units_num,keep_prob,forget_bias_init,lstm_layers_num,max_grad_norm
    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """ the constructor defines how the data flow of the model happens"""
        self._input = inputs
        batch_size = inputs.batch_size  # num of sequences
        time_steps = config.time_steps  # num of time steps every grad
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num  # num of elements in the hidden layer
        self.keep_prob = config.keep_prob
        ############# embedding layer #############
        with tf.variable_scope("embedding"):
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="embedding", shape=[vocab_size, units_num], dtype=tf.float32)
                b_embed_in = tf.get_variable(name="b_embed_in", shape=[units_num], dtype=tf.float32)
                embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            if is_training and config.keep_prob_state < 1:
                embedding_out = tf.nn.dropout(embedding, config.keep_prob)
            else:
                embedding_out = embedding



        ############# lstm layers #############

        # define basic lstm cell
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                num_units=units_num, forget_bias=config.forget_bias_init, state_is_tuple=True)

        possible_cell = lstm_cell
        # if dropout is needed add a dropout wrapper
        if is_training and config.keep_prob < 1:
            def possible_cell():
                if config.variational:
                    return dr.VariationalDropoutWrapperModified(lstm_cell(), batch_size, units_num,
                                                                output_keep_prob=config.keep_prob,
                                                                state_keep_prob=config.keep_prob_state)
                else:
                    return dr.DropoutWrapper(
                        lstm_cell(), output_keep_prob=config.keep_prob)


        # feed forward through lstm layers

        # build the lstm layers accordingly
        self.cell = []
        self._initial_state = []
        for i in range(config.lstm_layers_num):
            self.cell.append(possible_cell())
            self._initial_state.append(self.cell[i].zero_state(batch_size, dtype=tf.float32))

        # initiate outputs and states
        outputs = []
        state = []
        lstm_output = []
        for i in range(config.lstm_layers_num):
            outputs.append([])
            state.append([])

        with tf.variable_scope("lstm_layer_" + str(1)):
            state[0] = self._initial_state[0]

            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = self.cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, units_num]))

        for i in range(1, config.lstm_layers_num):
            with tf.variable_scope("lstm_layer_" + str(i + 1)):
                state[i] = self._initial_state[i]

                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = self.cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)

                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, units_num]))

        w = []
        b_embed_out = tf.get_variable(name="b_embed_out", shape=[vocab_size], dtype=tf.float32)
        logits = []

        # loss calculation
        loss = []
        self._cost = []
        for i in range(config.lstm_layers_num):
            # softmax layer
            with tf.variable_scope("loss" + str(i + 1)):
                w.append(tf.transpose(embedding_map))
                # w.append(tf.get_variable(name="w_softmax", shape=[units_num, vocab_size], dtype=tf.float32))
                # b.append(tf.get_variable(name="b_softmax", shape=[vocab_size], dtype=tf.float32))

                logits.append(tf.matmul(lstm_output[i], w[i]) + b_embed_out)
                loss.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [logits[i]], [tf.reshape(inputs.targets, [-1])],
                    [tf.ones([batch_size * time_steps], dtype=tf.float32)]))
                self._cost.append(tf.reduce_sum(loss[i]) / batch_size)
        cost = self._cost
        self._final_state = state

        # define training procedure for training data
        if not is_training:
            return

        # set learning rate as variable in order to anneal it throughout training
        self._lr = tf.Variable(0.0, trainable=False)

        # get trainable vars
        tvars = tf.trainable_variables()
        grads = []
        optimizer = []
        self._train_op = []
        for i in range(config.lstm_layers_num):
            with tf.name_scope("optimizer" + str(i + 1)):
                # apply grad clipping
                grad, _ = tf.clip_by_global_norm(tf.gradients(cost[i], tvars), config.max_grad_norm)
                grads.append(grad)
                # define optimizer
                optimizer.append(tf.train.GradientDescentOptimizer(self._lr))
                # define the train operation with the normalized grad
                self._train_op.append(optimizer[-1].apply_gradients(
                    zip(grads[-1], tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step()))
        with tf.name_scope("learning_rate"):
            # a placeholder to assign a new learning rate
            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")

            # function to update learning rate
            self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def update_masks(self, session):
        for i in range(config.lstm_layers_num):
            self.cell[i].update_masks(session)

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        for i in range(config.lstm_layers_num):
            self.cell[i].update_drop_params(session,
                                            output_keep_prob[i],
                                            state_keep_prob[i])

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    def cost(self, layer):
        return self._cost[layer]

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    def train_op(self, layer=-1):
        return self._train_op[layer]


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.time_steps = time_steps = config.time_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // time_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, time_steps, name=name)


def run_epoch(session, model, eval_op=None, verbose=True, layer=-1):
    """run the given model over its data"""

    start_time = time.time()
    costs = 0.0
    iters = 0

    # zeros initial state
    state = session.run(model.initial_state)
    if config.variational and eval_op is not None:
        model.update_masks(session)
    # determine the evaluations that are done every epoch
    fetches = {
        "cost": model.cost(layer),
        "final_state": model.final_state,
    }
    # if training is needed
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):

        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config(model_config_name):
    if model_config_name == "small":
        return SmallConfig()
    elif model_config_name == "medium":
        return MediumConfig()
    elif model_config_name == "large":
        return LargeConfig()
    elif model_config_name == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", model_config_name)


# class SmallConfig(object):
#   """Small config."""
#   init_scale = 0.1
#   learning_rate = 1.0
#   max_grad_norm = 5
#   num_layers = 2
#   num_steps = 20
#   hidden_size = 200
#   max_epoch = 4
#   max_max_epoch = 13
#   keep_prob = 1.0
#   lr_decay = 0.5
#   batch_size = 20
#   vocab_size = 10000


# batch_size,time_steps,vocab_size,units_num,keep_prob,forget_bias_init,lstm_layers_num,initializer,max_grad_norm

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 5
    time_steps = 35
    units_num = 1450
    max_max_epoch = 300
    keep_prob = 0.4
    keep_prob_state = 0.4
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    forget_bias_init = 0.0
    lstm_layers_num = 4
    init_epoch = 80
    sweeps = 1
    heuristic_num = 1
    variational = True
    moon = False
    drop_output = [[0.3, 0.0, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0], [0.4, 0.3, 0.2, 0.0], [0.5, 0.4, 0.3, 0.2]]
    drop_state = [[0.3, 0.0, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0], [0.4, 0.3, 0.2, 0.0], [0.5, 0.4, 0.3, 0.2]]

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 5.0
    max_grad_norm = 1
    time_steps = 35
    units_num = 1500
    max_max_epoch = 100
    keep_prob = 0.4
    keep_prob_state = 0.4
    lr_decay = 0.80
    batch_size = 20
    vocab_size = 10000
    forget_bias_init = 0.0
    lstm_layers_num = 2
    init_epoch = 10
    sweeps = 0
    heuristic_num = 1
    variational = True
    moon = False

simulation_name = "WTbiased_4layers_dp_3-4-4-5_3-3-4_2-3_2sdp_3-4-4-5_3-3-4_2-3_2variational_epoch_hlr1_hidden1450_gc5_lr1_dec8_epoch80"
model_config_name = "medium"
epoch_bias = 0
directory = "../results/" + simulation_name
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + '/train')
    os.makedirs(directory + '/test')
    os.makedirs(directory + '/saver')

data_path = "../PTB_data"
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, _ = raw_data

config = get_config(model_config_name)
eval_config = get_config(model_config_name)
eval_config.batch_size = 1
eval_config.num_steps = 35
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
with tf.name_scope("Train"):
    train_input = PTBInput(config=config, data=train_data, name="TrainInput")

    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, inputs=train_input)

    train_writer = tf.summary.FileWriter(directory + '/train',
                                         graph=tf.get_default_graph())
    tf.summary.scalar("Training Loss", m.cost(layer=-1))
    tf.summary.scalar("Learning Rate", m.lr)

with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)
    tf.summary.scalar("Validation Loss", mvalid.cost(layer=-1))

    valid_writer = tf.summary.FileWriter(directory + '/valid')

with tf.name_scope("Test"):
    test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, inputs=test_input)
    tf.summary.scalar("Test Loss", mtest.cost(layer=-1))
    test_writer = tf.summary.FileWriter(directory + '/test')

with tf.name_scope("summaries"):
    sv = tf.train.Supervisor(logdir=directory + '/saver')
    with sv.managed_session() as session:
        bestVal = config.vocab_size
        start_time = time.time()
        current_lr = None
        m.update_drop_params(session, config.drop_output[-1], config.drop_state[-1])

        # Gradual Training
        for sweep in range(config.sweeps):
            config.lr_decay = 0.8
            print("\n\nsweep #%d" % (sweep+1))
            for layer in range(0, config.lstm_layers_num):
                print("training layer #%d" % (layer + 1))
                valid_perplexity = []
                m.update_drop_params(session, config.drop_output[layer], config.drop_state[layer])
                heuristic_decay_counter = 1
                for i in range(0, config.init_epoch):
                    if i > 1:
                        if valid_perplexity[-1] > valid_perplexity[-2]:
                            if heuristic_decay_counter == config.heuristic_num:
                                current_lr = session.run(m.lr)
                                current_lr *= config.lr_decay
                                m.assign_lr(session, current_lr)
                                heuristic_decay_counter = 1
                            else:
                                heuristic_decay_counter += 1
                        if current_lr <= 0.5*config.learning_rate:
                            config.lr_decay = 0.8

                    elif i == 0:
                        current_lr = config.learning_rate
                        m.assign_lr(session, current_lr)

                    print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
                    train_perplexity = run_epoch(session, m, eval_op=m.train_op(layer), verbose=True, layer=layer)
                    train_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="train_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                simple_value=train_perplexity)])
                    train_writer.add_summary(train_sum, i + 1 + epoch_bias)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    valid_perplexity.append(run_epoch(session, mvalid, verbose=False, layer=layer))
                    valid_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="valid_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                simple_value=valid_perplexity[i])])
                    valid_writer.add_summary(valid_sum, i + 1 + epoch_bias)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity[i]))

                    if bestVal > valid_perplexity[-1]:
                        bestVal = valid_perplexity[-1]
                        test_perplexity = run_epoch(session, mtest, verbose=False, layer=layer)
                        test_sum = tf.Summary(
                            value=[tf.Summary.Value(tag="test_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                    simple_value=test_perplexity)])
                        test_writer.add_summary(test_sum, i + 1 + epoch_bias)
                        print("Epoch: %d Test Perplexity: %.3f" % (i + 1, test_perplexity))
            end_time = time.time()
            elapsed = end_time - start_time
            print("initialization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))


        # Traditional Training
        if current_lr is None:
            current_lr = config.learning_rate

        start_time = time.time()
        valid_perplexity = []
        heuristic_decay_counter = 1
        for i in range(config.max_max_epoch):
            if i > 1:
                if valid_perplexity[-1] > valid_perplexity[-2]:
                    if heuristic_decay_counter == config.heuristic_num:
                        current_lr *= config.lr_decay
                        heuristic_decay_counter = 1
                    else:
                        heuristic_decay_counter += 1
                    if current_lr <= 0.6*config.learning_rate:
                        config.lr_decay = 0.9
                        print("lr_decay was updated to %2.2f" % config.lr_decay)

            m.assign_lr(session, current_lr)

            print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op(-1), verbose=True)
            train_sum = tf.Summary(
                value=[tf.Summary.Value(tag="train_perplexity_end", simple_value=train_perplexity)])
            train_writer.add_summary(train_sum, i + 1 + epoch_bias)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity.append(run_epoch(session, mvalid, verbose=False))
            valid_sum = tf.Summary(
                value=[tf.Summary.Value(tag="valid_perplexity_end", simple_value=valid_perplexity[i])])
            valid_writer.add_summary(valid_sum, i + 1 + epoch_bias)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity[i]))

            test_perplexity = run_epoch(session, mtest, verbose=False)
            test_sum = tf.Summary(
                value=[tf.Summary.Value(tag="test_perplexity_end", simple_value=test_perplexity)])
            test_writer.add_summary(test_sum, i + 1 + epoch_bias)
            print("Epoch: %d Test Perplexity: %.3f" % (i + 1, test_perplexity))

        end_time = time.time()
        elapsed = end_time - start_time
        print("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
        if True:
            print("Saving model to %s." % (directory + '/saver'))
            sv.saver.save(session, directory + '/saver', global_step=sv.global_step)
