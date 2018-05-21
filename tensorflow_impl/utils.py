from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import ast
import re
import os
import logging

logger = logging.getLogger("logger")


def tvars_num():
    logger.info("model variables:")
    nvars = 0
    for var in tf.trainable_variables():
        logger.debug(var)
        shape = var.get_shape().as_list()
        nvars += np.prod(shape)

    logger.info('%2.2fM variables' % (nvars * 1e-6))

    return nvars


def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            if key == "drop_s" or key == "drop_h" or key == "max_update_norm":
                val = ast.literal_eval(val)

            setattr(config, key, val)

    return config


def get_config(config_pool, config_name):
    if config_name == "small":
        return config_pool.SmallConfig()
    if config_name == "mos":
        return config_pool.MosConfig()
    if config_name == "mos_gl":
        return config_pool.MosGL2Config()
    if config_name == "best":
        return config_pool.BestConfig()
    else:
        raise ValueError("Invalid model: %s", config_name)


def print_config(config):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    logger.info('\n' + '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))


def write_config(config, path):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    str = '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs)
    f = open(path, "w")
    f.write(str)
    f.close()


def get_simulation_name(args):
    waiver = ["cpu_device", "gpu_devices", "start_layer", "ckpt_file", "save", "verbose", "nonmono", "debug"]
    name = []
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None and key not in waiver:
            name.append(key + "-" + str(val).replace(",","-").replace(" ", "").replace("[", "").replace("]", ""))
    return "_".join(name)


def get_gpu_devices(str):
    devices_num = re.findall("[0-9]", str)
    return devices_num


def get_vars2restore(layer, model):
    if layer == 1:
        return None
    else:
        vars2load = []
        logger.debug("vars to restore when adding new layer")
        for var in tf.trainable_variables():
            logger.debug(var.op.name)
            if "embedding" in var.op.name:
                logger.debug("added")
                vars2load.append(var)
            if "layer_" in var.op.name:
                lstm_idx = re.findall("layer_([0-9])+", var.op.name)
                if int(lstm_idx[0]) < layer:
                    vars2load.append(var)
                    logger.debug("added")
            if "mos" in var.op.name and model.hid_size[layer-1] == model.hid_size[layer-2]:
                logger.debug("added")
                vars2load.append(var)
            if "w_embed_out" in var.op.name:
                logger.debug("added")
                vars2load.append(var)
            if "b_out" in var.op.name:
                logger.debug("added")
                vars2load.append(var)
        return vars2load


def write_to_summary(sum_path, config, train, valid, test):
    attr = sorted([attr for attr in dir(config) if not attr.startswith('__')])
    if not os.path.exists(sum_path):
        f = open("./summary.csv", "w")
        header = list()
        for arg in attr:
            header.append(arg)

        header.extend(["train_perp_tot", "valid_perp_tot", "test_perp_tot", "train_perp_0", "valid_perp_0", "test_perp_0"])

        f.write(",".join(header) +"\n")
    else:
        f = open("./summary.csv", "a")

    sum_list = list()
    for arg in attr:
        sum_list.append(str(getattr(config, arg)).replace(",","-").replace(" ", ""))

    scores = [[str(t), str(v), str(ts)] for t, v, ts in zip(train, valid, test)]
    sum_list.extend(scores[-1])
    scores.pop()
    scores = [str(item) for sublist in scores for item in sublist]
    sum_list.extend(scores)
    f.write(",".join(sum_list) +"\n")
    f.close()


def remove_tempstate_files(dir):
    for (folder, subs, files) in os.walk(dir):
        for filename in files:
            if "tempstate" in filename:
                file_path = os.path.join(dir, filename)
                yield(file_path)
