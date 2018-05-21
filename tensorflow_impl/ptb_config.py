class SmallConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    embed_init_scale = 0.1
    batch_size = 12
    bptt = 70
    layers = 2
    hid_size = [200] * layers
    embedding_size = 200
    vocab_size = 10000
    seed = 570164

    opt = "asgd"
    collect_stat = False

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = [0.05, 0.14, 0.15, 0.16]
    clip_by_layer = True
    epochs = 1

    GL = True
    DC = True
    AR = 2.0
    TAR = 1.0
    variational = 'batch'
    drop_e = 0.4
    drop_h = [0.225,0.4]
    drop_s = [0.5,0.5]
    drop_i = 0.1
    drop_embed_var = True
    wdecay = 1.2e-6
    drop_label = 0.1

    mos = True
    mos_experts_num = 15
    mos_drop = 0.29

    dynamic_eval = False
    dynamic_rms_step = True
    dynamic_rms_decay = True
    dynamic_decay = 0.025
    dynamic_lr = 1.
    dynamic_time_steps = 5
    dynamic_epsilon = 1e-8
    dynamic_clip_total_update = True



class MosConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 12
    time_steps = 70
    units_num = [960, 960, 620]
    embedding_size = 280
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = 0.25
    clip_by_layer = None
    layer_epoch = 1000
    entire_network_epoch = layer_epoch

    GL = False
    DC = False
    AR = 2.0
    TAR = 1.0
    variational = 'batch'
    keep_prob_embed = 0.6
    drop_output = [0.775,0.6]
    drop_state = [0.5,0.5]
    drop_i = 0.9
    drop_embed_var = True
    wdecay = 1.2e-6

    mos = True
    mos_context_num = 15
    mos_drop = 0.71

    dynamic_eval = True
    dynamic_rms_step = True
    dynamic_rms_decay = True
    dynamic_decay = 0.025
    dynamic_lr = 1.
    dynamic_time_steps = 5
    dynamic_epsilon = 1e-8
    dynamic_clip_total_update = True


class MosGL2Config(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    embed_init_scale = 0.1
    batch_size = 12
    bptt = 70
    hid_size = [850] * 3
    embedding_size = 280
    vocab_size = 10000
    layers = 3
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    collect_stat = False

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = 0.2 #[0.05, 0.16, 0.16, 0.16, 0.16]
    clip_by_layer = None
    epochs = 400

    GL = True
    DC = True
    AR = 2.0
    TAR = 1.0
    variational = 'batch'
    drop_e = 0.4
    drop_h = [0.225,0.4]
    drop_s = [0.5,0.5]
    drop_i = 0.1
    drop_embed_var = True
    wdecay = 1.2e-6
    drop_label = 0.1

    mos = True
    mos_experts_num = 15
    mos_drop = 0.29

    dynamic_eval = False
    dynamic_rms_step = True
    dynamic_rms_decay = True
    dynamic_decay = 0.025
    dynamic_lr = 1.
    dynamic_time_steps = 5
    dynamic_epsilon = 1e-8
    dynamic_clip_total_update = True

class MosGLConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    batch_size = 12
    time_steps = 70
    units_num = [850] * 3
    embedding_size = 280
    vocab_size = 10000
    lstm_layers_num = 3
    seed = 570164

    opt = "asgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"

    lr = 20.0
    lr_decay = 1.0
    max_update_norm = 0.25
    clip_by_layer = None
    layer_epoch = 400
    entire_network_epoch = layer_epoch

    GL = True
    DC = True
    AR = 2.0
    TAR = 1.0
    variational = 'batch'
    keep_prob_embed = 0.6
    drop_output = [0.775,0.6]
    drop_state = [0.5,0.5]
    drop_i = 0.9
    drop_embed_var = True
    wdecay = 1.2e-6
    shortcut = None

    mos = True
    mos_context_num = 15
    mos_drop = 0.71

    dynamic_eval = False
    dynamic_rms_step = True
    dynamic_rms_decay = True
    dynamic_decay = 0.025
    dynamic_lr = 1.
    dynamic_time_steps = 5
    dynamic_epsilon = 1e-8
    dynamic_clip_total_update = True

class BestConfig(object):
    init_scale = 0.0258198889747
    time_steps = 35
    batch_size = 20
    units_num = [1500] * 3
    embedding_size = 1500
    vocab_size = 10000
    seed = 570164

    opt = "sgd"
    opt_eps = 1e-5
    opt_inverse_type = "add"
    lr = 35.
    lr_decay = 0.85
    max_update_norm = 0.14
    clip_by_layer = False

    layer_epoch = 300
    entire_network_epoch = layer_epoch
    forget_bias_init = 0.0
    lstm_layers_num = 3
    GL = True
    DC = False
    AR = 2.0
    TAR = 1.0

    variational = 'epoch'
    keep_prob_embed = 0.4
    drop_output = [0.5, 0.27]
    drop_state = [0.5, 0.27]
    drop_embed_var = None
    drop_i = 0.9
    wdecay = 1.2e-6

    mos = False
    mos_context_num = 0
    mos_drop = 0.0



