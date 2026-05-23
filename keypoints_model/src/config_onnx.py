# MMDeploy export configuration for converting the trained HRNet-W32 pose
# estimation model to ONNX format for inference with ONNXRuntime.

# Specifies the inference backend that will run the exported model.
# 'onnxruntime' enables cross-platform CPU inference without a GPU dependency.
backend_config = dict(type='onnxruntime')

# Identifies the source codebase and task type so MMDeploy applies the
# correct export wrappers and pre/post-processing rewrite rules.
# export_postprocess_mask=False omits the segmentation mask postprocessing
# branch from the graph, keeping the export minimal for pose-only inference.
codebase_config = dict(
    type='mmpose',
    task='PoseDetection',
    export_postprocess_mask=False
)

# Controls the ONNX graph export behaviour.
#   export_params                : Embeds trained weights directly into the
#                                  .onnx file so it is fully self-contained.
#   keep_initializers_as_inputs  : False keeps weight tensors as internal
#                                  graph initializers rather than exposing
#                                  them as named inputs, reducing graph size.
#   opset_version                : ONNX operator set version 11 provides broad
#                                  compatibility across ONNXRuntime releases
#                                  while supporting all operators used by HRNet.
#   save_file                    : Output filename for the exported ONNX graph.
#   input_shape                  : Fixed (W, H) spatial dimensions of the model
#                                  input; must match codec['input_size'] = (288, 128)
#                                  defined in the training config.
#   input_names / output_names   : Named graph endpoints used to bind tensors
#                                  during ONNXRuntime inference sessions.
ir_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='best.onnx',
    input_shape=(288, 128),      # (W, H) — must match training codec input_size
    input_names=['input'],
    output_names=['output'],
)

# Describes the custom necklace keypoint dataset used during training.
# MMDeploy reads this block via Config.fromfile(cfg_file).dataset_info when
# building the pose task processor during ONNX export; it must be present in
# any config file referenced by metainfo=dict(from_file=...) in the
# train/val/test dataloaders.
#   dataset_name   : Arbitrary identifier for this custom dataset.
#   joint_weights  : Per-keypoint loss weights applied during training;
#                    both set to 1.0 for equal contribution.
#   keypoint_info  : Ordered dict mapping keypoint index → metadata.
#                      color  : BGR display colour used by the visualiser.
#                      id     : Must match the dict key.
#                      name   : Human-readable keypoint label.
#                      swap   : Paired keypoint name used for horizontal flip
#                               augmentation during training.
#                      type   : Body region tag ('upper' / 'lower').
#   paper_info     : Citation metadata for this dataset configuration.
#   sigmas         : Per-keypoint localisation standard deviations used by the
#                    COCO OKS metric; smaller values penalise localisation
#                    errors more strictly.
#   skeleton_info  : Ordered dict defining limb connections for visualisation.
#                      link   : Tuple of (source_name, target_name) keypoints.
#                      color  : BGR colour of the rendered skeleton edge.
dataset_info = dict(
    dataset_name='config_onnx',
    joint_weights=[1.0, 1.0],
    keypoint_info={
        0: dict(
            color=[255, 128, 0],
            id=0,
            name='neck_right',
            swap='neck_left',
            type='upper'
        ),
        1: dict(
            color=[0, 255, 128],
            id=1,
            name='neck_left',
            swap='neck_right',
            type='upper'
        )
    },
    paper_info=dict(
        author='anonim',
        container='MMPose',
        title='Necklace Keypoint Detection',
        year='2026'
    ),
    sigmas=[0.025, 0.025],
    skeleton_info={
        0: dict(
            color=[255, 255, 255],
            id=0,
            link=('neck_right', 'neck_left')
        )
    }
)