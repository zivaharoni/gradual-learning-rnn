import tensorflow as tf
import numpy as np


class DynamicEval(object):
    def __init__(self, config, tvars, grads):
        self._eps = config.dynamic_epsilon
        self._config = config
        self._tvars = tvars
        self._grads = grads

        self._step_size = config.dynamic_lr
        self._decay = config.dynamic_decay
        self._rms_step = config.dynamic_rms_step
        self._rms_decay = config.dynamic_rms_decay
        # self._max_grad_norm = config.max_grad_norm
        self._max_update_norm = config.max_update_norm

        # self._clipped_grads,_ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        self._clipped_grads = grads

        self._global_ms = []
        self._global_vars = []
        self._global_vars_assign_op = []
        self._decrate = []
        self._batches_num = tf.get_variable("batches_num",
                                            initializer=tf.zeros(shape=[], dtype=tf.float32),
                                            trainable=False)

        for var in tvars:
            self._global_vars.append(tf.get_variable(var.op.name + "_global",
                                                     initializer=tf.zeros_like(var, dtype=tf.float32),
                                                     trainable=False))
            self._global_ms.append(tf.get_variable(var.op.name + "_global_ms",
                                                     initializer=tf.zeros_like(var, dtype=tf.float32),
                                                     trainable=False))

            self._decrate.append(tf.get_variable(var.op.name + "_decrate",
                                                     initializer=tf.zeros_like(var, dtype=tf.float32),
                                                     trainable=False))

            with tf.name_scope("set_global_vars"):
                self._global_vars_assign_op.append(tf.assign(self._global_vars[-1], var))

    def accu_global_ms(self):
        accu = list()
        for grad, ms in zip(self._grads, self._global_ms):
            accu.append(tf.assign_add(ms, tf.square(grad)))
        inc_op = tf.assign_add(self._batches_num, tf.ones([]))
        return [accu, inc_op]

    def average_global_ms(self):
        ms_mean = [tf.assign(ms, tf.divide(ms, self._batches_num)) for ms in self._global_ms]
        return ms_mean

    def norm_ms_grads(self):
        g_sum = tf.reduce_mean([tf.reduce_mean(tf.sqrt(ms))  for ms in self._global_ms])
        ms_norm = [tf.assign(dec, tf.divide(tf.sqrt(ms), g_sum))for dec, ms in zip(self._decrate, self._global_ms)]

        return ms_norm


    def global_ms(self):
        return self._global_ms

    def batches_num(self):
        return self._batches_num

    def set_global_vars(self):
        return self._global_vars_assign_op

    def update_op(self):

        grads = list()
        for tvar, grad, ms, gvar, dec in zip(self._tvars, self._clipped_grads, self._global_ms, self._global_vars, self._decrate):
            grads.append(self.grad_update(grad, ms))

        if self._config.dynamic_clip_total_update is not None:
            grads, _ = tf.clip_by_global_norm(grads, self._config.max_update_norm)

        updates = list()
        eta = self._step_size
        lamb = self._decay
        for tvar, grad, ms, gvar, dec in zip(self._tvars, grads, self._global_ms, self._global_vars, self._decrate):
            delta = tf.multiply(-eta, grad) + tf.multiply(lamb, self.global_prior_decay(tvar, gvar, dec))
            updates.append(tf.assign_add(tvar, delta))

        return updates

    def grad_update(self, grad, ms):
        if self._rms_step is True:
            # if self._config.opt_inverse_type == "add":
            #     new_grad = grad / (ms + self._eps)
            # elif self._config.opt_inverse_type == "pseudo":
            #     condition = tf.greater_equal(ms, self._eps)
            #     new_grad = tf.where(condition, grad / ms, tf.zeros_like(ms))
            new_grad = grad / (ms + self._eps)
            # if self._config.dynamic_clip_total_update is not None:
            #     new_grad = tf.cond(tf.greater_equal(self._max_update_norm, tf.norm(new_grad)),
            #                        true_fn=lambda:new_grad,
            #                        false_fn=lambda:tf.divide(new_grad, tf.norm(new_grad))*self._max_update_norm)
        else:
            new_grad = grad

        return new_grad

    def global_prior_decay(self, tvar, gvar, dec):
        decay = gvar - tvar
        if self._rms_decay is True:
            rms_norm = tf.minimum(dec, 1/self._decay)
            decay *= rms_norm

        return decay


