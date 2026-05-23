# Dataset meta-information for the custom necklace keypoint detection task.
# This file is referenced by the training config via:
#   metainfo=dict(from_file='dataset_config.py')
# and inherited through:
#   _base_ = ['dataset_config.py']

dataset_info = dict(
    dataset_name='custom_necklace',

    paper_info=dict(
        author='anonim',
        title='Necklace Keypoint Detection',
        container='MMPose',
        year='2026'
    ),

    # Defines each keypoint's display name, index, visualisation color,
    # body region type, and its horizontal-flip counterpart (swap).
    # swap is used by RandomFlip to exchange left/right labels correctly
    # when the image is mirrored during training augmentation.
    keypoint_info={
        0: dict(
            name='neck_right',   # Right side of the necklace clasp
            id=0,
            color=[255, 128, 0], # Orange — displayed in visualisations
            type='upper',        # Upper-body keypoint category
            swap='neck_left'     # Paired with neck_left on horizontal flip
        ),
        1: dict(
            name='neck_left',    # Left side of the necklace clasp
            id=1,
            color=[0, 255, 128], # Green — displayed in visualisations
            type='upper',
            swap='neck_right'    # Paired with neck_right on horizontal flip
        ),
    },

    # Defines the skeleton edges drawn between keypoints during visualisation.
    # Each entry connects two keypoints by name and assigns a line color.
    # A single edge connects neck_right (id=0) to neck_left (id=1).
    skeleton_info={
        0: dict(
            link=('neck_right', 'neck_left'),
            id=0,
            color=[255, 255, 255]  # White connecting line
        )
    },

    # Per-keypoint loss weights used by KeypointMSELoss.
    # Both keypoints are weighted equally at 1.0; lower values down-weight
    # a keypoint's contribution to the total loss (e.g. for occluded points).
    joint_weights=[1.0, 1.0],

    # Per-keypoint OKS (Object Keypoint Similarity) sigma values.
    # sigma controls the spread of the Gaussian kernel in the OKS metric:
    #   smaller sigma → stricter localisation tolerance during evaluation.
    # 0.025 is tighter than the COCO default (e.g. 0.079 for shoulders),
    # reflecting the small and precise nature of necklace clasp keypoints.
    sigmas=[0.025, 0.025]
)