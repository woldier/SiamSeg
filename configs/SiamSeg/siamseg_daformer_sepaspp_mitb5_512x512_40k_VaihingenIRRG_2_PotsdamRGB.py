_base_ = [
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/uda/dcas.py',
    # dataset
    '../_base_/datasets/uda_vaihingenIRRG_2_potsdamRGB_512x512.py',
    # optimizer
    '../_base_/schedules/adamw.py',
    # schedule
    "../_base_/schedules/poly_schedule_40k.py",
    # runtime
    '../_base_/default_runtime.py'
]
work_dir = r'./result/{{fileBasenameNoExtension}}'
model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 1.25, 1.5, 1.5]
            # 给出class_weight
        )
    ),
    train_cfg=dict(work_dir=work_dir, cmap="isprs")
)
# dataset path set
# dataset samples_per_gpu 4 -> 1, workers_per_gpu 4 -> 3
# data_root = "/root/data/RSdata"  # custom dataset path
data = dict(
    samples_per_gpu=6, workers_per_gpu=3,
    # train=dict(source=dict(data_root=data_root), target=dict(data_root=data_root)),
    # val=dict(data_root=data_root),
    # test=dict(data_root=data_root),
)

# Modifications to Basic UDA
uda = dict(
    type='SiamSeg',
    # Increased Alpha
    alpha=0.999,
    # Pseudo-Label Crop
    # pseudo_weight_ignore_top=15,
    # pseudo_weight_ignore_bottom=120
    contras_model_cfg=dict(mod="target")
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    paramwise_cfg=dict(
        bypass_duplicate=True,  # 由于contrast model 中也用了backbone 会导致 报错 "ome parameters appear in more than one parameter group"
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)))
)

# learning policy
lr_config = dict(
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

checkpoint_config = dict(interval=500)
evaluation = dict(interval=500)
