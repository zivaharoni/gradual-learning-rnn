from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import reader
import os
import rnn_cell_additions as dr

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test", "model type. options are: GL, LAD, GL_LAD, Deep_LSTM_GL_LAD, Very_Deep_LSTM_GL_LAD, test")
flags.DEFINE_integer(
    "seed", 123456789, "random seed used for initialization")

FLAGS = flags.FLAGS


class PTBModel(object):
    """class for handling the ptb model"""

    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """the constructor builds the tensorflow graph"""
        self._input = inputs
        batch_size = inputs.batch_size  # num of sequences
        time_steps = config.time_steps  # num of time steps used in BPTT
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num    # num of units in the hidden layer
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # construct the embedding layer
        with tf.variable_scope("embedding"):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="embedding", shape=[vocab_size, units_num], dtype=tf.float32)
                b_embed_in = tf.get_variable(name="b_embed_in", shape=[units_num], dtype=tf.float32)
                embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            if is_training and config.keep_prob_embed < 1:
                embedding_out = tf.nn.dropout(embedding,
                                              config.keep_prob_embed)
            else:
                embedding_out = embedding

        # define basic lstm cell
        def lstm_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=units_num,
                                                forget_bias=config.forget_bias_init,
                                                state_is_tuple=True)
        possible_cell = lstm_cell
        # if dropout is needed add a dropout wrapper
        if is_training and config.drop_output is not None:
            def possible_cell():
                if config.variational is not None:
                    return dr.VariationalDropoutWrapper(lstm_cell(),
                                                        batch_size,
                                                        units_num)
                else:
                    return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(),
                                                         output_keep_prob=config.drop_output)

        # build the lstm layers and define their initial state
        self.cell = []
        self._initial_state = []
        for i in range(config.lstm_layers_num):
            self.cell.append(possible_cell())
            self._initial_state.append(self.cell[i].zero_state(batch_size, dtype=tf.float32))

        # organize layers' outputs and states in a list
        outputs = []
        state = []
        lstm_output = []
        for _ in range(config.lstm_layers_num):
            outputs.append([])
            state.append([])

        # unroll the cell to "time_steps" times
        # first lstm layer
        with tf.variable_scope("lstm_layer_" + str(1)):
            state[0] = self._initial_state[0]
            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = self.cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, units_num]))

        # rest of layers
        for i in range(1, config.lstm_layers_num):
            with tf.variable_scope("lstm_layer_" + str(i + 1)):
                state[i] = self._initial_state[i]
                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = self.cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)
                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, units_num]))

        # outer embedding bias
        b_embed_out = tf.get_variable(name="b_embed_out", shape=[vocab_size], dtype=tf.float32)
        # outer softmax matrix is tied with embedding matrix
        w_out = tf.transpose(embedding_map)

        # since using GL we have logits, losses and cost for every layer
        logits = []
        losses = []
        self._cost = []
        for i in range(config.lstm_layers_num):
            with tf.variable_scope("loss" + str(i + 1)):
                logits.append(tf.matmul(lstm_output[i], w_out) + b_embed_out)
                losses.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits[i]],
                                                                                 [tf.reshape(inputs.targets, [-1])],
                                                                                 [tf.ones([batch_size * time_steps],
                                                                                          dtype=tf.float32)]))
                self._cost.append(tf.reduce_sum(losses[i]) / batch_size)
        cost = self._cost
        self._final_state = state

        # define training procedure for training data
        if not is_training:
            return

        # set learning rate as variable in order to anneal it throughout training
        self._lr = tf.Variable(0.0, trainable=False)

        # get trainable vars
        tvars = tf.trainable_variables()

        # define an optimizer for every layer
        grads = []
        optimizer = []
        self._train_op = []
        for i in range(config.lstm_layers_num):
            with tf.name_scope("optimizer" + str(i + 1)):
                # apply grad clipping
                grad, _ = tf.clip_by_global_norm(tf.gradients(cost[i], tvars), config.max_grad_norm)
                grads.append(grad)
                # define optimizing method
                optimizer.append(tf.train.GradientDescentOptimizer(self._lr))
                # define the train operation with the normalized grad
                self._train_op.append(optimizer[-1].apply_gradients(
                    zip(grads[-1], tvars)))

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

    @property
    def global_step(self):
        return self._global_step


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
    # if variational every epoch --> update masks
    if config.variational == 'epoch' and eval_op is not None:
        model.update_masks(session)
    # determine the evaluations that are done every epoch
    fetches = {
        "cost": model.cost(layer),
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    for step in range(model.input.epoch_size):
        # if variational every batch --> update masks
        if config.variational == 'batch' and eval_op is not None:
            model.update_masks(session)
        # pass states between time batches
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


def train_optimizer(session, layer, m, mvalid, mtest, train_writer, valid_writer, test_writer):
    """ Trains the network by the given optimizer """
    global bestVal
    epochs_num = config.layer_epoch if layer != -1 else config.entire_network_epoch

    m.update_drop_params(session, config.drop_output[layer], config.drop_state[layer])
    current_lr = config.learning_rate
    m.assign_lr(session, current_lr)
    valid_perplexity = []
    for i in range(0, epochs_num):
        start_time = time.time()
        if len(valid_perplexity) > 2 and valid_perplexity[-1] > valid_perplexity[-2]:
            current_lr *= config.lr_decay
            m.assign_lr(session, current_lr)

        print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op(layer), verbose=True, layer=layer)
        train_sum = tf.Summary(value=[tf.Summary.Value(tag="train_perplexity_layer" + str(layer),
                                                       simple_value=train_perplexity)])
        train_writer.add_summary(train_sum, i + 1)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity.append(run_epoch(session, mvalid, verbose=False, layer=layer))
        valid_sum = tf.Summary(value=[tf.Summary.Value(tag="valid_perplexity_layer" + str(layer),
                                                       simple_value=valid_perplexity[-1])])
        valid_writer.add_summary(valid_sum, i + 1)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity[-1]))

        # evaluate test only when validation improves
        if bestVal > valid_perplexity[-1]:
            bestVal = valid_perplexity[-1]
            test_perplexity = run_epoch(session, mtest, verbose=False, layer=layer)
            test_sum = tf.Summary(value=[tf.Summary.Value(tag="test_perplexity_layer" + str(layer),
                                                          simple_value=test_perplexity)])
            test_writer.add_summary(test_sum, i + 1)
            print("Epoch: %d Test Perplexity: %.3f" % (i + 1, test_perplexity))
        elapsed = time.time() - start_time
        print("Epoch: %d took %02d:%02d" % (i + 1, elapsed // 60, elapsed % 60))


def get_config(model_config_name):
    if model_config_name == "LAD":
        return LADConfig()
    elif model_config_name == "GL":
        return GLConfig()
    elif model_config_name == "GL_LAD":
        return GLLADConfig()
    elif model_config_name == "Deep_LSTM_GL_LAD":
        return DeepGLLADConfig()
    elif model_config_name == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", model_config_name)


class LADConfig(object):
    """ Layerwise Adapted dropout config.
       The model obtains test perplexity of ~65.6 """
    init_scale = 0.04
    learning_rate = 1.0
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 35
    batch_size = 20
    units_num = 1500
    vocab_size = 10000
    entire_network_epoch = 160
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 2
    GL = False
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.0, 0.0], [0.5, 0.25]]
    drop_state = [[0.0, 0.0], [0.5, 0.25]]


class GLConfig(object):
    """ Gradual Learning config.
       The model obtains test perplexity of ~66.7 """
    init_scale = 0.04
    learning_rate = 1.0
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 35
    batch_size = 20
    units_num = 1500
    vocab_size = 10000
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 2
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.4, 0.0], [0.4, 0.4]]
    drop_state = [[0.4, 0.0], [0.4, 0.4]]


class GLLADConfig(object):
    """ Gradual Learning + Layerwise Adapted dropout config.
       The model obtains test perplexity of  ~64.5 """
    init_scale = 0.04
    learning_rate = 1.0
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 35
    batch_size = 20
    units_num = 1500
    vocab_size = 10000
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 2
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.3, 0.0], [0.5, 0.25]]
    drop_state = [[0.3, 0.0], [0.5, 0.25]]


class DeepGLLADConfig(object):
    """ 5-layered architecture that combines GL+LAD.
       The model obtains test perplexity of  ~61.7 """
    init_scale = 0.04
    learning_rate = 1.
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 25
    batch_size = 20
    units_num = 1500
    vocab_size = 10000
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 5
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0, 0.0],
                   [0.5, 0.4, 0.25, 0.0, 0.0], [0.5, 0.5, 0.4, 0.25, 0.0],
                   [0.5, 0.5, 0.5, 0.4, 0.25]]
    drop_state = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.4, 0.0, 0.0], [0.5, 0.5, 0.5, 0.4, 0.0],
                  [0.5, 0.5, 0.5, 0.5, 0.4]]


class TestConfig(object):
    """ for functionality check """
    init_scale = 0.04
    learning_rate = 1.0
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 35
    batch_size = 20
    units_num = 10
    vocab_size = 10000
    entire_network_epoch = 5
    layer_epoch = 5
    forget_bias_init = 0.0
    lstm_layers_num = 2
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.4, 0.0], [0.4, 0.4]]
    drop_state = [[0.4, 0.0], [0.4, 0.4]]

seed = FLAGS.seed
simulation_name = FLAGS.model + "_seed_" + str(seed)
model_config_name = FLAGS.model
directory = "./" + simulation_name
data_path = "./data"

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + '/train')
    os.makedirs(directory + '/test')
    os.makedirs(directory + '/saver')

raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, _ = raw_data

config = get_config(model_config_name)
eval_config = get_config(model_config_name)
eval_config.batch_size = 1
eval_config.num_steps = 35
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
            tf.summary.scalar("Training Loss", m.cost(layer=-1))
            tf.summary.scalar("Learning Rate", m.lr)
            tf.summary.scalar("global step", m.global_step)

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

        saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        session.run(tf.global_variables_initializer())
        # Gradual Training
        if config.GL:
            print("\ntraining gradually...\n")
            saver.save(session, directory + '/saver/model', global_step=0)
            for layer in range(0, config.lstm_layers_num-1):
                print("training layer #%d" % (layer + 1))
                start_time = time.time()
                train_optimizer(session, layer, m, mvalid, mtest, train_writer, valid_writer, test_writer)
                elapsed = time.time() - start_time
                print("optimization of layer %d took %02d:%02d:%02d\n" %
                      (layer + 1, elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
                # save model
                saver.save(session, directory + '/saver/model', global_step=layer+1)
        # Traditional Training
        m.update_drop_params(session, config.drop_output[-1], config.drop_state[-1])

        print("training the entire network...\n")
        start_time = time.time()
        train_optimizer(session, -1, m, mvalid, mtest, train_writer, valid_writer, test_writer)
        end_time = time.time()
        elapsed = end_time - start_time
        print("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
        # save model
        saver.save(session, directory + '/saver/model', global_step=config.lstm_layers_num)

        coord.request_stop()
        coord.join(threads)

    train_writer.close()
    valid_writer.close()
    test_writer.close()

if __name__ == "__main__":
    tf.app.run()
