# Config: HRNet-W32 + MSRAHeatmap
# Dataset: Custom Necklace (2 keypoints: neck_left, neck_right)
#
# Architecture : HRNet-W32 (TopdownPoseEstimator)
# Codec        : MSRAHeatmap  —  input (288×128), heatmap (72×32)
# Optimizer    : AdamW + AMP (Automatic Mixed Precision)
# LR Schedule  : LinearLR warmup → MultiStepLR decay (×0.1 at epoch 60 & 90)

# Inherits dataset meta-info (keypoint names, flip pairs, skeleton, etc.)
_base_ = ['dataset_config.py']

work_dir     = '../models/hrnet_necklace'
dataset_path = '../datasets/wednesday_last'


# 1. Runtime
# Training runs for 100 epochs; validation is triggered every 10 epochs.
max_epochs = 100

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10          # Run validation every 10 epochs
)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# All registry lookups (models, datasets, transforms) resolve under mmpose.
default_scope = 'mmpose'


# 2. Optimizer
# AmpOptimWrapper enables FP16 training via PyTorch AMP for faster
# throughput and lower VRAM usage. Dynamic loss scaling avoids
# gradient underflow without requiring a fixed scale factor.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0005,           # Base learning rate
        weight_decay=0.01    # L2 regularisation coefficient
    ),
    loss_scale='dynamic'     # Automatically adjust AMP loss scale
)


# 3. LR Scheduler
# Two-phase schedule:
#   Phase 1 — LinearLR warmup over the first 100 iterations:
#              LR rises from (0.1 × base_lr) to base_lr to avoid
#              unstable gradients at the start of training.
#   Phase 2 — MultiStepLR decay: LR is multiplied by gamma (0.1)
#              at epoch 60 and again at epoch 90, reducing it from
#              base_lr → base_lr×0.1 → base_lr×0.01.
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=100,
        start_factor=0.1,    # Start LR = base_lr × 0.1
        by_epoch=False       # Steps counted in iterations, not epochs
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        milestones=[60, 90], # Epochs at which LR is multiplied by gamma
        gamma=0.1,           # LR reduction factor at each milestone
        by_epoch=True
    )
]


# 4. Codec: MSRAHeatmap
# Defines how keypoint annotations are encoded into Gaussian heatmaps
# for training, and decoded back to coordinates during inference.
#   input_size   : (W, H) fed into the model backbone.
#   heatmap_size : (W, H) of the generated Gaussian target maps;
#                  stride = input / heatmap = 288/72 = 4.
#   sigma        : Standard deviation of the Gaussian kernel (pixels
#                  in heatmap space). Smaller σ → sharper, harder targets.
#   unbiased     : Enables unbiased Gaussian encoding (DARK decoding)
#                  for sub-pixel accuracy during coordinate decoding.
codec = dict(
    type='MSRAHeatmap',
    input_size=(288, 128),
    heatmap_size=(72, 32),
    sigma=1.5,
    unbiased=True,
)


# 5. Model
# TopdownPoseEstimator: crops a person bounding box, runs the backbone
# to extract features, then predicts per-keypoint heatmaps via the head.
model = dict(
    type='TopdownPoseEstimator',

    # Normalises pixel values to zero-mean/unit-variance using
    # ImageNet statistics and converts BGR (OpenCV) to RGB.
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),

    # HRNet-W32: maintains high-resolution feature maps throughout by
    # running parallel multi-resolution branches and fusing them repeatedly.
    # Width (num_channels) doubles at each new branch: 32 → 64 → 128 → 256.
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            # Stage 1: single-branch bottleneck stem (64 channels).
            stage1=dict(
                num_modules=1, num_branches=1, block='BOTTLENECK',
                num_blocks=(4,), num_channels=(64,)),
            # Stage 2: two branches (32, 64 channels); first multi-scale fusion.
            stage2=dict(
                num_modules=1, num_branches=2, block='BASIC',
                num_blocks=(4, 4), num_channels=(32, 64)),
            # Stage 3: three branches (32, 64, 128); repeated 4x for richer fusion.
            stage3=dict(
                num_modules=4, num_branches=3, block='BASIC',
                num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            # Stage 4: four branches (32, 64, 128, 256); final high-res output
            #          taken from the 32-channel branch for the head.
            stage4=dict(
                num_modules=3, num_branches=4, block='BASIC',
                num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256))
        ),
        # Initialise from the official HRNet-W32 ImageNet pretrained weights.
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
                       'pretrain_models/hrnet_w32-36af842e.pth'
        ),
    ),

    # HeatmapHead: 1x1 conv projects the 32-channel backbone output to
    # out_channels (= number of keypoints) heatmaps.
    # deconv_out_channels=None skips transposed-conv upsampling because
    # the backbone already outputs at the target heatmap resolution.
    head=dict(
        type='HeatmapHead',
        in_channels=32,          # Must match the highest-resolution HRNet branch
        out_channels=2,          # One heatmap per keypoint (neck_left, neck_right)
        deconv_out_channels=None,# No extra upsampling; heatmap stride = 4 is sufficient
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True,  # Down-weights invisible/occluded keypoints
            loss_weight=1.0
        ),
        # Decoder converts predicted heatmap peaks back to (x, y) coordinates.
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(288, 128),
            heatmap_size=(72, 32),
            sigma=1.5,
            unbiased=True,       # Must match codec settings
        )
    ),

    # Flip-test averages predictions on the original and horizontally flipped
    # image for more stable and accurate keypoint localisation at inference.
    test_cfg=dict(
        flip_test=True,
        shift_heatmap=True,      # Corrects systematic offset introduced by flipping
    )
)


# 6. Pipelines
# Training pipeline applies geometric and photometric augmentations to
# improve generalisation. Each dict is an MMPose transform called in order.
train_pipeline = [
    dict(type='LoadImage'),

    # Computes the bounding-box centre and scale from the annotation bbox,
    # used downstream by TopdownAffine to define the crop region.
    dict(type='GetBBoxCenterScale'),

    # Random horizontal flip; left/right keypoint labels are swapped
    # automatically using the flip_pairs defined in dataset_config.py.
    dict(type='RandomFlip', direction='horizontal'),

    # Randomly perturbs the bounding box with translation, scale jitter,
    # and in-plane rotation to simulate pose and viewpoint variation.
    dict(type='RandomBBoxTransform',
         shift_factor=0.16,
         scale_factor=[0.6, 1.4],  # Tighter range than [0.5, 1.5] to limit extreme crops
         rotate_factor=45),         # Up from 40° for wider rotation coverage

    # Warps the crop into a fixed (288x128) canvas via an affine transform.
    # use_udp=True applies the Unbiased Data Processing alignment convention
    # for more precise coordinate mapping between image and heatmap space.
    dict(type='TopdownAffine',
         input_size=codec['input_size'],
         use_udp=True),

    # Photometric augmentations via the Albumentations library.
    # Applied after affine warp so augmentations act on the final crop.
    dict(type='Albumentation',
         transforms=[
             # Spatial blurring to simulate motion or focus blur.
             dict(type='Blur', p=0.1),
             dict(type='MedianBlur', p=0.1),

             # Randomly erases rectangular patches to simulate occlusion.
             # fill_value matches ImageNet mean to minimise distribution shift.
             dict(type='CoarseDropout',
                  max_holes=6,                    # Up from 4 for more occlusion variety
                  max_height=0.25,                # Up from 0.2; larger occluded area
                  max_width=0.25,                 # Up from 0.2
                  fill_value=[124, 116, 104],     # Approximate ImageNet mean (BGR)
                  p=0.4),                         # Up from 0.3

             # Simulates changes in lighting conditions.
             dict(type='RandomBrightnessContrast',
                  brightness_limit=0.3,           # ±30% brightness variation
                  contrast_limit=0.3,             # ±30% contrast variation
                  p=0.4),                         # Up from 0.3

             # Simulates white-balance and saturation differences across cameras.
             dict(type='HueSaturationValue',
                  hue_shift_limit=15,             # Hue jitter range (degrees)
                  sat_shift_limit=25,             # Saturation jitter range
                  p=0.3),                         # Up from 0.2

             # Adds pixel-level Gaussian noise to simulate sensor noise.
             dict(type='GaussNoise',
                  var_limit=(10.0, 40.0),         # Noise variance range
                  p=0.2),
         ]
    ),

    # Encodes (x, y) keypoint coordinates into Gaussian heatmap targets
    # using the codec settings defined in section 4.
    dict(type='GenerateTarget', encoder=codec),

    # Collates image, heatmap targets, keypoint weights, and meta-info
    # into the format expected by the model's forward pass.
    dict(type='PackPoseInputs')
]

# Validation pipeline applies only deterministic transforms (no augmentation)
# to ensure reproducible and comparable evaluation metrics.
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine',
         input_size=codec['input_size'],
         use_udp=True),              # Must match train_pipeline setting
    dict(type='PackPoseInputs')
]


# 7. Data Loaders
# batch_size=8 with num_workers=2 and persistent_workers avoids the
# overhead of re-spawning worker processes between epochs.
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,         # Keep workers alive across epochs
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_path,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(from_file='dataset_config.py'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,                 # Evaluate every sample; do not drop remainder
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=dataset_path,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=dict(from_file='dataset_config.py'),
        pipeline=val_pipeline,
    )
)

# Reuse the val dataloader for the test split.
test_dataloader = val_dataloader


# 8. Evaluators
# CocoMetric computes standard COCO keypoint metrics:
# AP, AP50, AP75, APm, APl, AR — primary metric is coco/AP.
val_evaluator = dict(
    type='CocoMetric',
    ann_file=dataset_path + '/annotations/val.json'
)
test_evaluator = val_evaluator


# 9. Visualizer
# PoseLocalVisualizer overlays predicted skeleton and keypoints on images.
# LocalVisBackend writes visualisation outputs to the work_dir.
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)


# 10. Hooks
# Hooks execute automatically at predefined training lifecycle events.
default_hooks = dict(
    # Logs iteration time to monitor throughput bottlenecks.
    timer=dict(type='IterTimerHook'),

    # Writes loss and metric values to the log every 50 iterations.
    logger=dict(type='LoggerHook', interval=50),

    # Steps the LR schedulers defined in param_scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # Saves a checkpoint every 10 epochs; keeps only the 5 most recent.
    # Also saves a separate 'best' checkpoint whenever coco/AP improves.
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',         # Metric to monitor for best-model saving
        rule='greater',              # Higher coco/AP is better
        max_keep_ckpts=5             # Limit disk usage to last 5 checkpoints
    ),

    # Synchronises the random seed across distributed workers each epoch
    # to ensure consistent shuffling when using multiple GPUs.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # Renders and saves keypoint visualisations during validation.
    visualization=dict(type='PoseVisualizationHook', enable=True),
)


# 11. Env
# cudnn_benchmark=False: safer for variable-resolution inputs; set True
#   only if all input shapes are fixed (trades stability for speed).
# mp_start_method='fork': faster worker startup on Linux; use 'spawn'
#   on Windows or when CUDA is initialised before DataLoader workers.
# backend='nccl': NCCL is the recommended collective communication
#   backend for multi-GPU training on NVIDIA hardware.
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)