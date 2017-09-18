import numpy as np

class TestConfig(object):
    """ 5-layered architecture that combines GL+LAD.
       The model obtains test perplexity of  ~61.7 """
    units_num = 1150
    embedding_size = 400
    init_scale = 1.0 / np.sqrt(units_num)
    emb_init_scale = 0.1
    lr = 30.0
    lr_decay = 1.0
    lwgc_grad_norm = [0.08,0.18,0.22]
    max_grad_norm = [0.25]
    time_steps = 35
    batch_size = 40
    vocab_size = 10000
    layer_epoch = 300
    entire_network_epoch = layer_epoch
    forget_bias_init = 0.0
    lstm_layers_num = 2
    LWGC = False
    GL = False
    DC = False
    AR = 2.0
    TAR = 1.0
    restore = False
    restore_path = ''
    variational = 'epoch'
    opt = 'asgd'
    keep_prob_embed = 0.9
    drop_output = [[0.3, 0.0], [0.6, 0.6]]
    drop_state = [[0.3, 0.0], [0.7, 0.7]]

class LWGCConfig(object):
    """ 5-layered architecture that combines GL+LAD.
       The model obtains test perplexity of  ~61.7 """
    units_num = 1500
    embedding_size = units_num
    init_scale = 1.0 / np.sqrt(units_num)
    emb_init_scale = 0.05
    lr = 1.0
    lr_decay = 0.9
    lwgc_grad_norm = [1.5,2.0,5.0]
    max_grad_norm = [0.25]
    time_steps = 35
    batch_size = 30
    vocab_size = 10000
    layer_epoch = 300
    entire_network_epoch = layer_epoch
    forget_bias_init = 0.0
    lstm_layers_num = 2
    LWGC = True
    GL = True
    DC = False
    AR = 2.0
    TAR = 1.0
    restore = False
    restore_path = ''
    variational = 'epoch'
    opt = "asgd"
    keep_prob_embed = 0.4
    drop_output = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0, 0.0],
                   [0.5, 0.4, 0.25, 0.0, 0.0], [0.5, 0.5, 0.4, 0.25, 0.0],
                   [0.5, 0.5, 0.5, 0.4, 0.25]]
    drop_state = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.4, 0.0, 0.0], [0.5, 0.5, 0.5, 0.4, 0.0],
                  [0.5, 0.5, 0.5, 0.5, 0.4]]

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
    learning_rate = 1.0
    lr_decay = 0.85
    max_grad_norm = 3.5
    time_steps = 25
    batch_size = 20
    units_num = 1500
    vocab_size = 10000
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 5
    GL = True
    restore = False
    restore_path = 'D:\Machine_learning\ptb\gradual-learning-rnn/trained_model/batch90/model-1'
    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0, 0.0],
                   [0.5, 0.4, 0.25, 0.0, 0.0], [0.5, 0.5, 0.4, 0.25, 0.0],
                   [0.5, 0.5, 0.5, 0.4, 0.25]]
    drop_state = [[0.3, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.4, 0.0, 0.0], [0.5, 0.5, 0.5, 0.4, 0.0],
                  [0.5, 0.5, 0.5, 0.5, 0.4]]

