class Config:
    # data
    x_dim = 2
    y_dim = 1

    # model
    model_type = "mlp"
    num_layers = 2
    hidden_dim = 64
    activation = "gelu"
    act_normalization = "batch"
    ckpt_path = None

    # diffusion
    n_T = 500
    betas = (1e-4, 0.02)
    drop_prob = 0.1
    ws_test = [0.0, 0.5, 2.0]

    # training
    n_epoch = 100
    batch_size = 128
    num_workers = 0
    lrate = 1e-3
    weight_decay = 1e-6
    save_dir = "./data/branin_1/"
    validate = False
