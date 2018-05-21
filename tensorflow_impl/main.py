from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import wiki2_config
import os
import sys
import argparse
import logging

from model import PTBModel
import ptb_reader
import ptb_config
import utils


def build_graph(config, is_training):
    def build_model(data, init, is_training, reuse):
        with tf.variable_scope("Model", reuse=reuse, initializer=init):
            data = ptb_reader.PTBInput(config=config, data=data)
            model = PTBModel(is_training=is_training, config=config, data=data)
        writer = tf.summary.FileWriter(directory + '/train',
                                       graph=tf.get_default_graph())

        return model, writer

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale, seed=seed)

    with tf.name_scope("Train"):
        train_model, train_writer = \
            build_model(train_data, initializer, is_training=is_training, reuse=None)
        logger.debug("train shape: (%d,%d), train len: %d, epoch size: %d" %
                     (train_model.data.data.shape[0], train_model.data.data.shape[1], train_model.data.data_len,
                      train_model.data.epoch_size))

    with tf.name_scope("Valid"):
        valid_model, valid_writer = \
            build_model(valid_data, initializer, is_training=None, reuse=True)
        logger.debug("valid shape: (%d,%d), valid len: %d, epoch size: %d" %
                     (valid_model.data.data.shape[0], valid_model.data.data.shape[1], valid_model.data.data_len,
                      valid_model.data.epoch_size))

    test_model, test_writer = None, None
    if not is_training:
        with tf.name_scope("Test"):
            test_model, test_writer = \
                build_model(test_data, initializer, is_training=None, reuse=True)
            logger.debug("test shape: (%d,%d), test len: %d, epoch size: %d" %
                         (valid_model.data.data.shape[0], valid_model.data.data.shape[1], valid_model.data.data_len,
                          valid_model.data.epoch_size))

    saver = tf.train.Saver(var_list=tf.trainable_variables())
    restore_saver = None
    if config.GL:
        vars2load = utils.get_vars2restore(config.layers, train_model)
        if vars2load is not None:
            restore_saver = tf.train.Saver(var_list=vars2load)

    config.tvars_num = '%fM' % (utils.tvars_num() * 1e-6)

    return train_model, train_writer, valid_model, valid_writer, test_model, test_writer, saver, restore_saver

def train_epoch(session, model, verbose=True):
    """run the given model over its data"""
    start_time = time.time()

    model.data.shuffle()

    if args.collect_stat:
        min_update = 1e20
        max_update = 0
        mean_update = 0

    losses = 0.0
    iters = 0

    # zeros initial state
    state = session.run(model.initial_state)

    feed_dict_masks = {}
    # if variational every epoch --> update masks
    if config.variational is not None:
        # generate masks for LSTM and mos if exists
        feed_dict_masks = model.gen_masks(session)

        # generate mask for weight-dropped LSTM
        if config.DC:
            feed_dict_masks.update(model.gen_wdrop_mask(session))

        # generate mask for embedding
        if config.drop_i > 0.0 or config.drop_e > 0.0:
            feed_dict_masks.update(model.gen_emb_mask(session))

            # randomize words to drop from the vocabulary
            if config.drop_i > 0.0:
                words2drop = list()
                for i in range(config.batch_size):
                    rand = np.random.rand(config.vocab_size)
                    bin_vec = np.zeros(config.vocab_size, dtype=np.int32)
                    bin_vec[rand < config.drop_i] = 1
                    drop = np.where(bin_vec == 1)
                    words2drop.append(drop[0])
                dropped = list()

    # evaluate loss and final state for all devices
    fetches = {
        "loss": model.loss,
        "final_state": model.final_state
    }

    # perform train op if training
    fetches["eval_op"] = model.train_op
    if args.collect_stat:
        fetches["stat"] = model.stat_ops

    for step in range(model.data.epoch_size):
        # pass states between time batches
        feed_dict = dict(feed_dict_masks.items())
        for j, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[j].c
            feed_dict[h] = state[j].h

        feed_dict.update(model.data.get_batch(step*model.data.bptt))

        if config.drop_i > 0.0:
            for i, batch in enumerate(feed_dict[model.data.input_data]):
                for j, w in enumerate(batch):
                    if w in words2drop[i]:
                        dropped.append(w)
                        feed_dict[model.emb_mask][i, j, :] = 0

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        state = vals["final_state"]

        losses += loss / (1-config.drop_label)
        iters += 1 * (1-config.drop_label)

        if args.collect_stat:
            stat = vals["stat"][0]
            min_update = np.minimum(min_update, stat)
            max_update = np.maximum(max_update, stat)
            mean_update += stat

        if verbose and step % (model.data.epoch_size // 10) == 10:
            logger.info("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                        (step * 1.0 / model.data.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                         iters * model.data.batch_size * model.data.bptt / (time.time() - start_time)))

        if config.variational == 'epoch' and config.drop_embed_var is None:
            feed_dict_masks.update(model.gen_emb_mask(session))

        # if variational every batch --> update masks
        if config.variational == 'batch':
            # generate masks for LSTM and mos if exists
            feed_dict_masks.update(model.gen_masks(session))

            # generate mask for embedding
            if config.drop_i > 0.0 or config.drop_e > 0.0:
                feed_dict_masks.update(model.gen_emb_mask(session))

                # randomize words to drop from the vocabulary
                if config.drop_i > 0.0:
                    words2drop = list()
                    for i in range(config.batch_size):
                        rand = np.random.rand(config.vocab_size)
                        bin_vec = np.zeros(config.vocab_size, dtype=np.int32)
                        bin_vec[rand < config.drop_i] = 1
                        drop = np.where(bin_vec == 1)
                        words2drop.append(drop[0])

        # generate mask for weight-dropped LSTM
        if config.DC:
            feed_dict_masks.update(model.gen_wdrop_mask(session))

    if config.drop_i > 0.0:
        logger.info("dropped %d/%d words" % (len(dropped), model.data.data_len))

    if args.collect_stat:
        logger.info("mean update: %2.2f, min update: %2.2f, max update: %2.2f" %
                    (mean_update /iters , min_update, max_update))
    return np.exp(losses / iters)


def evaluate(session, model, verbose=True):
    """run the given model over its data"""
    start_time = time.time()

    losses = 0.0
    iters = 0

    # zeros initial state
    state = session.run(model.initial_state)

    # evaluate loss and final state for all devices
    fetches = {
        "loss": model.loss,
        "final_state": model.final_state
    }

    for step in range(model.data.epoch_size):
        # pass states between time batches
        feed_dict = dict()
        for j, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[j].c
            feed_dict[h] = state[j].h

        feed_dict.update(model.data.get_batch(step*model.data.bptt))

        vals = session.run(fetches, feed_dict)

        loss = vals["loss"]
        state = vals["final_state"]

        losses += loss
        iters += 1

        if verbose and step % (model.data.epoch_size // 10) == 10:
            logger.info("%.3f perplexity: %.3f bits: %.3f speed: %.0f wps" %
                        (step * 1.0 / model.data.epoch_size, np.exp(losses / iters), np.log2(np.exp(losses / iters)),
                         iters * model.data.batch_size * model.data.bptt / (time.time() - start_time)))
    return np.exp(losses / iters)


def train_optimizer(session, layer, m, mvalid, train_writer, valid_writer, saver):
    """ Trains the network by the given optimizer """

    bestVal = m._vocab_size

    def stop_criteria(epoch):
        stop_window = 10
        if epoch == epochs_num - 1:
            return True

        if len(valid_perplexity) > stop_window:

            if np.min(valid_perplexity[:-stop_window]) < valid_perplexity[-1]:
                return True
            else:
                return False
        else:
            return False

    epochs_num = config.epochs

    logger.info("updating dropout probabilities")
    m.update_drop_params(session, config.drop_h, config.drop_s)

    if args.save:
        logger.info("saving initial model....\n")
        save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
        logger.info("save path is: %s" % save_path)

    lr_decay = config.lr_decay
    current_lr = session.run(m.lr)

    if config.opt == "asgd" or config.opt == "arms" or config.opt == "masgd" or config.opt == "marms":
        nonmono = args.nonmono

    valid_perplexity = []
    i = session.run(m.epoch)
    should_stop = False
    while not should_stop:
        start_time = time.time()
        if len(valid_perplexity) >= 2 and valid_perplexity[-1] > valid_perplexity[-2]:
            current_lr *= lr_decay
            m.assign_lr(session, current_lr)
            lr_sum = tf.Summary(value=[tf.Summary.Value(tag="learning_rate_track" + str(layer),
                                                        simple_value=current_lr)])
            train_writer.add_summary(lr_sum, i + 1)
            if (config.opt == "asgd" or config.opt == "arms" or
                config.opt == "masgd" or config.opt == "marms") \
                    and not session.run(m.optimizer.trigger):
                if len(valid_perplexity) > nonmono and np.min(valid_perplexity[:-nonmono]) < valid_perplexity[-1]:
                    logger.info("setting trigger and T")
                    m.optimizer.set_trigger(session)
                    m.optimizer.set_T(session, i)
                    lr_decay = 1.0

        logger.info("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

        if config.opt == "asgd" or config.opt == "arms" or (config.opt == "marms" or config.opt == "masgd"):
            logger.info("Trigger is %s, T=%d" % (bool(session.run(m.optimizer.trigger)), session.run(m.optimizer.T)))

        ###################################### train ######################################
        train_perplexity = train_epoch(session, m, verbose=True)
        train_sum = tf.Summary(value=[tf.Summary.Value(tag="train_perplexity_layer" + str(layer),
                                                       simple_value=train_perplexity)])
        train_writer.add_summary(train_sum, i + 1)
        logger.info("Epoch: %d Train Perplexity: %.3f Bits: %.3f " % (i + 1, train_perplexity, np.log2(train_perplexity)))

        if ((config.opt == "asgd" or config.opt == "arms" or config.opt == "marms" or config.opt == "masgd")
                and session.run(m.optimizer.trigger)):
            logger.info("saving model weights....")
            session.run(m.optimizer.save_vars)
            logger.info("setting averaged weights....")
            session.run(m.optimizer.final_assign_op)

        ###################################### valid ######################################
        valid_perplexity.append(evaluate(session, mvalid, verbose=False))
        valid_sum = tf.Summary(value=[tf.Summary.Value(tag="valid_perplexity_layer" + str(layer),
                                                       simple_value=valid_perplexity[-1])])
        valid_writer.add_summary(valid_sum, i + 1)
        logger.info("Epoch: %d Valid Perplexity: %.3f Bits: %.3f" % (i + 1, valid_perplexity[-1], np.log2(valid_perplexity[-1])))

        # save model only when validation improves
        if bestVal > valid_perplexity[-1]:
            bestVal = valid_perplexity[-1]
            if args.save:
                try:
                    save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
                    # print("save path is: %s" % save_path)
                    logger.info("save path is: %s" % save_path)
                except:
                    pass

        if (config.opt == "asgd" or config.opt == "arms" or config.opt == "marms" or config.opt == "masgd") \
                and not session.run(m.optimizer.trigger):
            should_stop = False if i != epochs_num - 1 else True
        else:
            should_stop = stop_criteria(i)

        if ((config.opt == "asgd" or config.opt == "arms" or config.opt == "marms" or config.opt == "masgd")
                and session.run(m.optimizer.trigger)):
            logger.info("loading model weights....")
            session.run(m.optimizer.load_vars)


        elapsed = time.time() - start_time
        logger.info("Epoch: %d took %02d:%02d\n" % (i + 1, elapsed // 60, elapsed % 60))
        i = m.epoch_inc(session)


    save_path = saver.save(session, directory + '/saver/best_model' + str(layer))
    logger.info("save path is: %s" % save_path)

    if args.save:
        logger.info("restoring best model of current optimizer....")
        saver.restore(session, directory + '/saver/best_model' + str(layer))
    m.epoch_reset(session)


def test(session, m, mvalid, mtest):
    """ Trains the network by the given optimizer """

    start_time = time.time()
    logger.info("train:")
    train_perplexity = evaluate(session, m)
    logger.info("valid:")
    valid_perplexity = evaluate(session, mvalid)
    logger.info("test:")
    test_perplexity = evaluate(session, mtest)
    logger.info("Train Perplexity: %.3f Bits: %.3f " % (train_perplexity, np.log2(train_perplexity)))
    logger.info("Valid Perplexity: %.3f Bits: %.3f " % (valid_perplexity, np.log2(valid_perplexity)))
    logger.info("Test Perplexity: %.3f Bits: %.3f " % (test_perplexity, np.log2(test_perplexity)))

    elapsed = time.time() - start_time
    logger.info("Evaluation took %02d:%02d" % (elapsed // 60, elapsed % 60))

    return  train_perplexity, valid_perplexity, test_perplexity


def main():
    try:
        logger.info("training model...")
        ###################################### GL configs and restore ######################################
        tf.set_random_seed(seed)
        np.random.seed(seed)
        hid_size = config.hid_size[:]
        start_layer = 0 if config.GL else config.layers - 1
        if args.start_layer is not None:
            if args.ckpt_file is None:
                raise ValueError("must provide ckpt_file flag with start_layer_flag")
            print("\nstarting training from layer: %d\n" % args.start_layer)
            start_layer = args.start_layer - 1

        ###################################### GL training ######################################
        start_time_total = time.time()
        for layer in range(start_layer, config.layers):
            config.layers = layer + 1
            config.hid_size = hid_size[:layer+1]

            ###################################### build graph ######################################
            with tf.Graph().as_default() as graph:
                train_model, train_writer, valid_model, valid_writer, _, _, saver, restore_saver = \
                    build_graph(config, is_training=True)

            ###################################### train ######################################
            with tf.Session(graph=graph, config=sess_config) as session:
                session.run(tf.global_variables_initializer())
                config.hid_size = hid_size[:layer + 1]

                if restore_saver is not None and config.GL:
                    model_path = "%s/saver/best_model%d" % (directory, layer-1)
                    logger.info("loading from %s" % model_path)
                    restore_saver.restore(session, model_path)

                logger.info("training layer #%d" % (layer + 1))

                start_time = time.time()
                train_optimizer(session, layer, train_model, valid_model, train_writer, valid_writer, saver)
                elapsed = time.time() - start_time

                logger.info("optimization of layer %d took %02d:%02d:%02d\n" %
                            (layer + 1, elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

            tf.reset_default_graph()

            train_writer.close()
            valid_writer.close()

        elapsed = time.time() - start_time_total
        logger.info("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

    except KeyboardInterrupt:
        print("\n\n")
        logger.info("training was exited by user...")

    ###################################### GL evaluation ######################################
    batch_size = config.batch_size
    config.batch_size = 1
    train_perplexity, valid_perplexity, test_perplexity = [], [], []
    for layer in range(start_layer, config.layers):
        config.layers = layer + 1
        config.hid_size = hid_size[:layer + 1]
        logger.info("Evaluating layer %d" % (layer+1))

        ###################################### build graph ####################################
        with tf.Graph().as_default() as graph:
            train_model, train_writer, valid_model, valid_writer, test_model, test_writer, saver, _ = \
                build_graph(config, is_training=False)


        ###################################### evaluation ######################################
        with tf.Session(graph=graph, config=sess_config) as session:
            session.run(tf.global_variables_initializer())
            model_path = directory + '/saver/best_model' + str(config.layers - 1)
            saver.restore(session, model_path)
            train_pp, valid_pp, test_pp = test(session, train_model, valid_model, test_model)
            train_perplexity.append(train_pp)
            valid_perplexity.append(valid_pp)
            test_perplexity.append(test_pp)


    config.batch_size = batch_size
    utils.write_to_summary("./summary.csv", config, train_perplexity, valid_perplexity, test_perplexity)

    ############################ close handlers ##################################
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    for file_path in utils.remove_tempstate_files(directory + '/saver'):
        os.remove(file_path)


if __name__ == "__main__":
    ###################################### argument parsing ######################################
    ap = argparse.ArgumentParser()

    ap.add_argument("--gpu_devices",        type=str, default="0", help="gpu device list")
    ap.add_argument("--cpu_device",         type=str, default="0", help="cpu device")
    ap.add_argument("--ckpt_file",          type=str, default=None, help="file path for restore")
    ap.add_argument("--start_layer",        type=int, default=None, help="train from layer")
    ap.add_argument("--name",               type=str, default="debug", help="simulation name")
    ap.add_argument("--model",              type=str, default="small", help="model name")
    ap.add_argument("--seed",               type=int, default=None, help="seed")
    ap.add_argument("--data",               type=str, default="ptb", help="data type")
    ap.add_argument("--opt",                type=str, default=None, help="optimizer name")
    ap.add_argument("--opt_eps",            type=float, default=None, help="optimizer epsilon")
    ap.add_argument("--opt_inverse_type",   type=str, default=None, help="optimizer inversion type")
    ap.add_argument("--opt_clip_by_var",    dest='opt_clip_by_var', action='store_true', help="optimizer clip update by vars or globally")
    ap.add_argument("--opt_mom",            type=float, default=None, help="optimizer momentum")
    ap.add_argument("--opt_mom_decay",      type=float, default=None, help="optimizer momentum decay")
    ap.add_argument("--lr",                 type=float, default=None, help="training learning rate")
    ap.add_argument("--lr_decay",           type=float, default=None, help="learning rate decay")
    ap.add_argument("--max_update_norm",    type=float, default=None, help="max update norm")
    ap.add_argument("--batch_size",         type=int, default=None, help="batch size")
    ap.add_argument("--time_steps",         type=int, default=None, help="bptt truncation")
    ap.add_argument("--hid_size",           type=int, default=None, help="#of units in lstm layer")
    ap.add_argument("--embedding_size",     type=int, default=None, help="#of units in embedding")
    ap.add_argument("--epochs",             type=int, default=None, help="epochs per layer")
    ap.add_argument("--layers",             type=int, default=None, help="#of lstm layers")
    ap.add_argument("--nonmono",            type=int, default=5, help="non monotonic trigger for asgd")
    ap.add_argument("--AR",                 type=float, default=None, help="activation regularization parameter")
    ap.add_argument("--TAR",                type=float, default=None, help="temporal activation regularization parameter")
    ap.add_argument("--drop_output",        type=str, default=None, help="list of dropout parameters for outer connections")
    ap.add_argument("--drop_state",         type=str, default=None, help="list of dropout parameters for recurrent connections")
    ap.add_argument("--keep_prob_embed",    type=float, default=None, help="keep prob for embedding")
    ap.add_argument("--opt_c_lipsc",        type=float, default=None, help="for lwgc")
    ap.add_argument("--drop_i",             type=float, default=None, help="drop words")
    ap.add_argument("--mos_drop",           type=float, default=None, help="drop mos")
    ap.add_argument("--mos_context_num",    type=int, default=None, help="#of experts")
    ap.add_argument("--wdecay",             type=float, default=None, help="weight decay")
    ap.add_argument("--mos",                dest='mos', action='store_true')
    ap.add_argument("--no_eval",            dest='no_eval', action='store_true')
    ap.add_argument("--GL",                 dest='GL', action='store_false')
    ap.add_argument("--DC",                 dest='DC', action='store_true')
    ap.add_argument('--verbose',            dest='verbose', action='store_true')
    ap.add_argument('--save',               dest='save', action='store_true')
    ap.add_argument('--collect_stat',       dest='collect_stat', action='store_true')
    ap.add_argument('--drop_embed_var',     dest='drop_embed_var', action='store_true')
    ap.add_argument('--clip_by_layer',      dest='clip_by_layer', action='store_true')
    ap.add_argument('--trig',               dest='trig', action='store_true')
    ap.add_argument('--debug',              dest='debug', action='store_true')

    ap.add_argument('--finetune',           dest='finetune', action='store_true')

    ap.set_defaults(trig=None)
    ap.set_defaults(debug=None)
    ap.set_defaults(finetune=None)
    ap.set_defaults(collect_stat=None)
    ap.set_defaults(clip_by_layer=None)
    ap.set_defaults(mos=None)
    ap.set_defaults(no_eval=None)
    ap.set_defaults(drop_embed_var=None)
    ap.set_defaults(opt_clip_by_var=None)
    ap.set_defaults(GL=None)
    ap.set_defaults(DC=None)
    ap.set_defaults(verbose=None)
    ap.set_defaults(save=True)
    args = ap.parse_args()

    ###################################### general configs ######################################
    if args.data == "ptb":
        config_pool = ptb_config
    elif args.data == "wiki2":
        config_pool = wiki2_config
    else:
        raise ValueError("Invalid database")

    config = utils.get_config(config_pool, args.model)
    config = utils.read_flags(config, args)

    if args.seed is not None:
        config.seed = seed = args.seed
    elif config.seed == 0:
        config.seed = seed = np.random.randint(0, 1000000)
    else:
        seed = config.seed

    simulation_name = utils.get_simulation_name(config)
    if args.ckpt_file is None:
        directory = "./results/" + simulation_name
    else:
        directory = "/".join(args.ckpt_file.split("/")[:-2])

    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory + '/train')
        os.makedirs(directory + '/test')
        os.makedirs(directory + '/saver')
    elif args.ckpt_file is None and not args.save:
        raise ValueError("simulation already exists; rerun with name flag")

    sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                 inter_op_parallelism_threads=2,
                                 intra_op_parallelism_threads=8)

    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = ",".join(utils.get_gpu_devices(args.gpu_devices))

    ###################################### define logger ###################################################
    logFormatter = logging.Formatter("%(asctime)-15s | %(levelname)-8s | %(message)s")
    logger = logging.getLogger("logger")

    if args.debug is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    if args.finetune is None:
        fileHandler = logging.FileHandler("{0}/logger.log".format(directory))
    else:
        fileHandler = logging.FileHandler("{0}/logger-finetune{1}.log".format(directory, config.layers - 1))

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if args.verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    logger.info("cmd line: python " + " ".join(sys.argv))
    ###################################### read data ###################################################

    data_path = "./data"

    raw_data = ptb_reader.ptb_raw_data(data_path, args.data)
    train_data, valid_data, test_data, _ = raw_data

    logger.info("Simulation configurations" )
    utils.print_config(config)

    try:
        sys.exit(main())
    except Exception:
        logger.exception(Exception)
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
