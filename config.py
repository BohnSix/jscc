import torch
import datetime
import torch.nn as nn


class config:

    train_data_dir = ["/data/bohnsix/datasets/DIV2K/DIV2K_train_HR"]
    test_data_dir = ["/data/bohnsix/datasets/kodak"]
    batch_size = 4
    num_workers = 8

    print_step = 225
    plot_step = 1000
    logger = None

    # training details
    image_dims = (3, 256, 256)
    lr = 1e-4
    aux_lr = 1e-3
    distortion_metric = "MSE"  # 'MS-SSIM'

    use_side_info = False
    train_lambda = 64
    eta = 0.2

    channel = {"type": "awgn", "chan_param": 10}
    multiple_rate = [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
    ]
    ga_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 256, 256, 256],
        depths=[1, 1, 2, 4],
        num_heads=[8, 8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )

    gs_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 256, 256, 256],
        depths=[4, 2, 1, 1],
        num_heads=[8, 8, 8, 8],
        window_size=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )

    fe_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=256,
        depths=[4],
        num_heads=[8],
        window_size=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        rate_choice=multiple_rate,
    )

    fd_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=256,
        depths=[4],
        num_heads=[8],
        window_size=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        rate_choice=multiple_rate,
    )

    # EHB-NTSCC settings
    ehb_mode = True
    ehb_num_experts = 4
    ehb_use_red_loss = True
    ehb_lambda_red = 1e-4
    ehb_rate_hidden_dim = 64
    ehb_target_cbr = 0.08
    ehb_lambda_cbr = 10.0
