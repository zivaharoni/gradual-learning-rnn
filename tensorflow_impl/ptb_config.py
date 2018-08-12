class MosGLConfig(object):
    forget_bias_init = 0.0
    init_scale = 0.04
    embed_init_scale = 0.1
    batch_size = 40
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



