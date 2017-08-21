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
