# Nuovo modello per includere resnet18, aggiunto da Gionni e MaryGio
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,  # cambia se hai un numero diverso di classi
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # auxiliary head (opzionale, puoi rimuoverla se non usata)
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
