class Config:
    # data
    x_dim = 24 * 6
    y_dim = 25 * 2
    subset = None

    # model
    model_type = "mlp"
    num_layers = 4
    hidden_dim = 512
    activation = "gelu"
    act_normalization = "batch"
    ckpt_path = None

    # diffusion
    n_T = 500
    betas = (1e-4, 0.02)
    drop_prob = 0.1
    ws_test = [0.0, 0.5, 2.0]

    # training
    n_epoch = None
    batch_size = None
    num_workers = 8
    lrate = None
    weight_decay = 1e-6
    save_dir = None
    validate = False

    joint_importance = {
        0: 6,
        1: 4,
        2: 4,
        3: 5,
        4: 3,
        5: 3,
        6: 5,
        7: 2,
        8: 2,
        9: 5,
        10: 1,
        11: 1,
        12: 3,
        13: 4,
        14: 4,
        15: 2,
        16: 4,
        17: 4,
        18: 3,
        19: 3,
        20: 2,
        21: 2,
        22: 1,
        23: 1,
    }
