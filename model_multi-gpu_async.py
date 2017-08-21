from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import ptb_reader
import enwik8_reader
import ptb_config
import enwik8_config
import os
import rnn_cell_additions as dr

flags = tf.flags
logging = tf.logging

# flags for distributed tf
flags.DEFINE_integer(
    "gpu_num", 2, "#of gpus")

# flags for error handling
flags.DEFINE_string(
    "ckpt_file", "", "name of checkpoint file to load model from")
flags.DEFINE_integer(
    "start_layer", 0, "if restore is needed, mention the layer to start from")

# flags for model
flags.DEFINE_string(
    "model", "parallel", "model type. options are: GL, LAD, GL_LAD, Deep_LSTM_GL_LAD, test")
flags.DEFINE_string(
    "data", "enwik8", "dataset. options are: ptb, enwik8")
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
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num  # num of units in the hidden layer
        self._gpu_num = config.gpu_num  # #og gpu used in model

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
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            # the embedding matrix is allocated in the cpu to save valuable gpu memory for the model.
            embedding_map = tf.get_variable(
                name="embedding", shape=[vocab_size, units_num], dtype=tf.float32)
            b_embed_in = tf.get_variable(name="b_embed_in", shape=[units_num], dtype=tf.float32)
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
                with tf.device("/gpu:%d" % i), tf.name_scope("gpu-%d" % i):
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
            optimizer = []
            self._train_op = []
            for j in range(self._gpu_num):
                gpu_optimizers = []
                for i in range(config.lstm_layers_num):
                    gpu_optimizers.append(tf.train.GradientDescentOptimizer(self._lr))
                    self._train_op.append(gpu_optimizers[i].apply_gradients(
                        zip(all_grads[j][i], tvars), global_step=self._global_step))
                optimizer.append(gpu_optimizers)

    def reduce_loss(self, all_loss):
        """ average the loss obtained by gpus

        Args:
            grads: 2D array, the [i,j] element stands for the loss of the j-th layer of the i-th gpu

        Returns:
            grads: a list of the loss for each layer
        """

        total_loss = []
        for i in range(config.lstm_layers_num):
            if self._gpu_num > 1:
                layer_loss = [all_loss[j][i] for j in range(self._gpu_num)]
                total_loss.append(tf.reduce_mean(layer_loss))
            else:
                total_loss.append(all_loss[0][i])
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
            cell.append(possible_cell())
            initial_state.append(cell[0].zero_state(batch_size, dtype=tf.float32))
            state[0] = initial_state[0]
            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, units_num]))

        # rest of layers
        for i in range(1, config.lstm_layers_num):
            with tf.variable_scope("lstm%d" % (i + 1)):
                cell.append(possible_cell())
                initial_state.append(cell[i].zero_state(batch_size, dtype=tf.float32))
                state[i] = initial_state[i]
                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)
                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, units_num]))

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

        for i in range(config.lstm_layers_num):
            with tf.variable_scope("loss" + str(i + 1)):
                logits.append(tf.matmul(lstm_output[i], w_out) + b_embed_out)
                losses.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits[i]],
                                                                                 [tf.reshape(targets, [-1])],
                                                                                 [tf.ones([batch_size * time_steps],
                                                                                          dtype=tf.float32)]))
                loss.append(tf.reduce_sum(losses[i]) / batch_size)

                grad, _ = tf.clip_by_global_norm(tf.gradients(loss[i], tvars), config.max_grad_norm)
                grads.append(grad)
        final_state = state

        return loss, grads, cell, initial_state, final_state

    def initial_state(self, device_num):
        return self._initial_state[device_num]

    def final_state(self, device_num):
        return self._final_state[device_num]

    def loss(self, layer=-1):
        return self._loss[layer]

    def train_op(self, layer=-1, gpu=0):
        return self._train_op[gpu][layer]

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

    def next_layer_ptr(self, session, next_layer):
        session.run(self._layer_ptr_update, feed_dict={self._new_layer: next_layer})

    def update_masks(self, session):
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
                self._cell[j][i].update_masks(session)

    def update_drop_params(self, session, output_keep_prob, state_keep_prob):
        for j in range(self._gpu_num):
            for i in range(config.lstm_layers_num):
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


def run_epoch(session, model, eval_op=None, verbose=True, layer=-1):
    """run the given model over its data"""

    start_time = time.time()
    losses = 0.0
    iters = 0

    # zeros initial state for all devices
    state = []
    for k in range(model.gpu_num):
        state.append(session.run(model.initial_state(k)))

    # if variational every epoch --> update masks
    if config.variational == 'epoch' and eval_op is not None:
        model.update_masks(session)

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
            model.update_masks(session)

        # pass states between time batches
        feed_dict = {}
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
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(losses / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(losses / iters)


def train_optimizer(session, layer, m, mvalid, mtest, train_writer, valid_writer, test_writer, saver):
    """ Trains the network by the given optimizer """
    global bestVal
    epochs_num = config.layer_epoch if layer != -1 else config.entire_network_epoch

    m.update_drop_params(session, config.drop_output[layer], config.drop_state[layer])
    if FLAGS.ckpt_file is "":
        current_lr = config.learning_rate
        m.assign_lr(session, current_lr)
    else:
        current_lr = session.run(m.lr)

    valid_perplexity = []
    i = session.run(m.epoch)
    while current_lr > 8e-3:
        start_time = time.time()
        if len(valid_perplexity) >= 2 and valid_perplexity[-1] > valid_perplexity[-2]:
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


def get_config(model_config_name):
    if model_config_name == "parallel":
        return config_pool.ParallelConfig()
    elif model_config_name == "batched":
        return config_pool.BatchedConfig()
    elif model_config_name == "test":
        return config_pool.TestConfig()
    else:
        raise ValueError("Invalid model: %s", model_config_name)


if FLAGS.data == 'ptb':
    config_pool = ptb_config
    reader = ptb_reader.ptb_raw_data
elif FLAGS.data == 'enwik8':
    config_pool = enwik8_config
    reader = enwik8_reader.enwik8_raw_data
else:
    raise ValueError("Invalid data: %s", FLAGS.data)

seed = FLAGS.seed
simulation_name = FLAGS.data + "_" + FLAGS.model + "_seed_" + str(seed)
model_config_name = FLAGS.model
directory = "./results/" + simulation_name
data_path = "./data"

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + '/train')
    os.makedirs(directory + '/test')
    os.makedirs(directory + '/saver')

raw_data = reader(data_path)
train_data, valid_data, test_data, _ = raw_data

config = get_config(model_config_name)
config.gpu_num = FLAGS.gpu_num

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

        saver = tf.train.Saver(var_list=tf.trainable_variables())

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=sess_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        if FLAGS.start_layer > 0:
            saved_model_path = directory + '/saver/' + FLAGS.ckpt_file
            print("\nloading model from: " + saved_model_path)
            saver.restore(session, saved_model_path)
            start_layer = FLAGS.start_layer - 1
        else:
            session.run(tf.global_variables_initializer())
            start_layer = 0

        # Gradual Training
        if config.GL:
            print("\ntraining gradually...\n")
            for layer in range(start_layer, config.lstm_layers_num-1):
                print("training layer #%d" % (layer + 1))
                start_time = time.time()
                train_optimizer(session, layer, m, mvalid, _, train_writer, valid_writer, _, saver)
                elapsed = time.time() - start_time
                print("optimization of layer %d took %02d:%02d:%02d\n" %
                      (layer + 1, elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
                # save model
                saver.save(session, directory + '/saver/model', global_step=layer+1)

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
