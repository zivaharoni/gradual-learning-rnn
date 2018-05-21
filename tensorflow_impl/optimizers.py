from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import logging
import re

logger = logging.getLogger("logger")

def clip_by_layer(updates, clip_norms):
    clipped_updates = list()
    for update, tvar in zip(updates, tf.trainable_variables()):
        logger.debug("optimizers-clip_by_layer: var name is %s" % tvar.op.name)
        if "embedding" in tvar.op.name:
            clip_norm = clip_norms[0]
        elif "layer_" in tvar.op.name:
            depth = int(re.findall("layer_([0-9])+", tvar.op.name)[0])
            clip_norm = clip_norms[depth]
        elif "mos" in tvar.op.name or "out" in tvar.op.name:
            clip_norm = clip_norms[-1]
        else:
            raise ValueError("clip by layer: depth was not selected")

        logger.debug("clipping " + tvar.op.name + " by " + str(clip_norm))
        clipped_updates.append(tf.clip_by_norm(update, clip_norm))

    return clipped_updates


class SGD(object):
    def __init__(self, model, grads, tvars, config):

        if config.collect_stat:
            model.stat_ops.append(tf.add_n([tf.square(tf.norm(g)) for g in grads]))

        self._updates = grads
        self._optimizer = tf.train.GradientDescentOptimizer(model.lr)

        if config.max_update_norm is not None:
            if config.clip_by_layer:
                self._updates = clip_by_layer(self._updates, config.max_update_norm)
            else:
                self._updates, _ = tf.clip_by_global_norm(self._updates, config.max_update_norm)

        self._train_op = self._optimizer.apply_gradients(
                zip(self._updates, tvars), global_step=model.global_step)


    @property
    def train_op(self):
        return self._train_op

    @property
    def updates(self):
        return self._updates

    @property
    def optimizer(self):
        return self._optimizer


class ASGD(object):

    def __init__(self, model, grads, tvars, config):

        sgd = SGD(model, grads, tvars, config)

        # self._model = model
        self._updates = sgd.updates

        with tf.name_scope("trigger"):
            self._trigger = tf.get_variable("ASGD_trigger", initializer=tf.constant(False, dtype=tf.bool), trainable=False)
            self._set_trigger = tf.assign(self._trigger, True)

            self._T = tf.get_variable("T", initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            self._new_T = tf.placeholder(tf.int32, shape=[], name="new_T")
            self._set_T = tf.assign(self._T, self._new_T)

        self._train_op = list()
        self._train_op.append(sgd.train_op)

        with tf.name_scope("averaging"):
            self._save_vars = []
            self._load_vars = []
            self._final_vars = []
            self._final_assign_op = []

            for var in tvars:
                self._final_vars.append(tf.get_variable(var.op.name + "final",
                                                        initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))

                with tf.name_scope("final_average"):
                    cur_epoch_num = tf.cast((model.epoch - self._T + 1) * model.data.epoch_size, dtype=tf.float32)
                    self._final_assign_op.append(tf.assign(var, self._final_vars[-1] / cur_epoch_num))

                with tf.name_scope("assign_current_weights"):
                    tmp_var = (tf.get_variable(var.op.name + "tmp",
                                               initializer=tf.zeros_like(var, dtype=tf.float32), trainable=False))
                    self._save_vars.append(tf.assign(tmp_var, var))
                    self._load_vars.append(tf.assign(var, tmp_var))

        with tf.name_scope("trigger_mux"):
            def trigger_on():
                with tf.name_scope("trigger_is_on"):
                    op = list()
                    op.append(tf.identity(self._trigger))
                    op.append(tf.identity(self._T))
                    for i, var in enumerate(tvars):
                        op.append(tf.assign_add(self._final_vars[i], var))

                return op

            def trigger_off():
                with tf.name_scope("trigger_is_off"):
                    op = list()
                    op.append(tf.identity(self._trigger))
                    op.append(tf.identity(self._T))
                    for i, var in enumerate(tvars):
                        op.append(tf.identity(self._final_vars[i]))

                return op

            with tf.control_dependencies(self._train_op):
                self._train_op.append(tf.cond(self._trigger, lambda: trigger_on(), lambda: trigger_off()))


    @property
    def train_op(self):
        return self._train_op

    @property
    def trigger(self):
        return self._trigger

    @property
    def T(self):
        return self._T

    @property
    def final_assign_op(self):
        return self._final_assign_op

    @property
    def save_vars(self):
        return self._save_vars

    @property
    def load_vars(self):
        return self._load_vars

    def set_trigger(self, session):
        return session.run(self._set_trigger)

    def set_T(self, session, T):
        return session.run(self._set_T, feed_dict={self._new_T: T})


###################################### ADD TO RMS OPTIMIZER ################################################
############################################################################################################
#####            if args.collect_stat:                                                            ##########
#####                 self._stat_ops.append(tf.add_n([tf.square(tf.norm(g)) for g in grads]))     ##########
############################################################################################################

# TODO: Update and create arms
class RMSprop(object):
    def __init__(self, model, grads, tvars, decay=0.9, use_opt=True): #TODO:replace with 0.9
        self._decay = decay
        self._config = model.config
        self._eps = model.config.opt_eps
        self._max_update_norm = model.config.max_update_norm
        self._lr = model.lr

        self._grads = grads
        self._tvars = tvars

        self._ms = []
        self._ms_accu_op = []
        for tvar, g in zip(tvars, grads):
            self._ms.append(tf.get_variable(g.op.name + "_ms",
                                            initializer=tf.ones_like(tvar, dtype=tf.float32) / 50,
                                            trainable=False))

            g = tf.convert_to_tensor(g)
            with tf.name_scope("set_global_vars"), tf.control_dependencies([g]):
                self._ms_accu_op.append(tf.assign(self._ms[-1],
                                                  self._decay * self._ms[-1] + (1 - self._decay) * tf.square(g)))

        self._updates = list()
        self._train_op = list()

        # if self._config.opt_inverse_type == "add":
        #     logger.info("inversion stability by adding epsilon")
        # elif self._config.opt_inverse_type == "pseudo":
        #     logger.info("inversion stability by thresholding eigen-values by epsilon")

        # compute updates
        with tf.control_dependencies(self._ms_accu_op):
            for grad, ms in zip(grads, self._ms):
                self._updates.append(self.update(grad, ms))

        # clip updates
        if self._max_update_norm is not None:
            logger.info("clipping total update")
            self._updates, _ = tf.clip_by_global_norm(self._updates, self._max_update_norm)

        # apply updates op
        if use_opt:
            for i, tvar in enumerate(tvars):
                delta = tf.multiply(-self._lr, self._updates[i])
                self._train_op.append(tf.assign_add(tvar, delta))
            # self._train_op.extend(self._ms_accu_op)
        else:
            self._train_op = None


    def update(self, grad, ms):
        if self._config.opt_inverse_type == "add":
            update = grad / (ms + self._eps)
        elif self._config.opt_inverse_type == "pseudo":
            condition = tf.greater_equal(ms, self._eps)
            update = tf.where(condition, grad / ms, grad)
        else:
            raise ValueError("opt_inverse_type has invalid value")

        return update

    @property
    def updates(self):
        return self._updates

    @property
    def train_op(self):
        return self._train_op

    @property
    def grads(self):
        return self._grads

    def get_grads_norm(self):
        g_norm = []
        for grad in self._grads:
            g_norm.append(tf.reduce_sum(tf.square(grad)))

        ms_norm = []
        for ms in self._ms:
            ms_norm.append(tf.reduce_min(ms))

        u_norm = []
        for update in self._updates:
            u_norm.append(tf.reduce_sum(tf.square(update)))
        return tf.sqrt(tf.add_n(g_norm)), tf.reduce_min(tf.stack(ms_norm)), tf.sqrt(tf.add_n(u_norm))

    def get_ms_max(self):
        ms_max = []
        for ms in self._ms:
            ms_max.append(tf.reduce_max(ms))

        return tf.reduce_max(tf.stack(ms_max))

    def get_ms(self):
        ms = []
        for m in self._ms:
            m = tf.convert_to_tensor(m)
            ms.append(tf.reshape(m,shape=[-1]))

        return tf.concat(ms, 0)

    def get_grad(self):
        gs = []
        for g in self._grads:
            g = tf.convert_to_tensor(g)
            gs.append(tf.reshape(g,shape=[-1]))

        return tf.concat(gs, 0)

