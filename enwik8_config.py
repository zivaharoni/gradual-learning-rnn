class DeepGLLADConfig(object):
    """ 4-layered architecture that combines GL+LAD.
       The model obtains test perplexity of  ???? """
    init_scale = 0.05
    learning_rate = 0.2
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 50
    batch_size = 128
    units_num = 1125
    vocab_size = 205
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 4
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.9
    drop_output = [[0.9, 0.0, 0.0, 0.0], [0.92, 0.87, 0.0, 0.0], [0.92, 0.9, 0.8, 0.0], [0.92, 0.9, 0.9, 0.85]]
    drop_state = [[0.9, 0.0, 0.0, 0.0], [0.92, 0.87, 0.0, 0.0], [0.92, 0.9, 0.8, 0.0], [0.92, 0.9, 0.9, 0.85]]


class BatchedConfig(object):
    """ 4-layered architecture that combines GL+LAD.
       The model obtains test perplexity of  ???? """
    init_scale = 0.05
    learning_rate = 0.05
    lr_decay = 0.8
    max_grad_norm = 5
    time_steps = 50
    batch_size = 300
    units_num = 1000
    vocab_size = 205
    entire_network_epoch = 80
    layer_epoch = 80
    forget_bias_init = 0.0
    lstm_layers_num = 4
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.9
    drop_output = [[0.9, 0.0, 0.0, 0.0], [0.92, 0.87, 0.0, 0.0], [0.92, 0.9, 0.8, 0.0], [0.92, 0.9, 0.9, 0.85]]
    drop_state = [[0.9, 0.0, 0.0, 0.0], [0.92, 0.87, 0.0, 0.0], [0.92, 0.9, 0.8, 0.0], [0.92, 0.9, 0.9, 0.85]]


class TestConfig(object):
    """ for functionality check """
    init_scale = 0.05
    learning_rate = .3
    lr_decay = 0.85
    max_grad_norm = 5
    time_steps = 50
    batch_size = 128
    units_num = 10
    vocab_size = 205
    entire_network_epoch = 2
    layer_epoch = 1
    forget_bias_init = 0.0
    lstm_layers_num = 2
    GL = True
    variational = 'epoch'
    keep_prob_embed = 0.9
    drop_output = [[0.9, 0.0], [0.9, 0.9]]
    drop_state = [[0.9, 0.0], [0.9, 0.9]]
