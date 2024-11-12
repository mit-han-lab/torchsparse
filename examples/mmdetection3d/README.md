# TorchSparse for MMDetection3D Plugin Demo

This tutorial demonstrates how to evaluate TorchSparse integrated MMDetection3D models. Follow the steps below to install dependencies, configure paths, convert model weights, and run the demo.

## Dependencies

1. **MMDetection3D Installation**: Follow the [MMDetection3D documentation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).
2. **Dataset Preparation**: Pre-process the datasets as described [here](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html).
3. **TorchSparse Installation**: Install [TorchSparse](https://github.com/mit-han-lab/torchsparse).
4. **Install TorchSparse Plugin for MMDetection3D**:
    1. Clone this repository.
    2. Navigate to `examples/mmdetection3d` and run `pip install -v -e .`.

## Notes

- For model evaluation, change the data root in the original MMDetection3D's model config to the full path of the corresponding dataset root.

## Steps

1. Install the dependencies.
2. Specify the base paths and model registry.
3. **IMPORTANT,** Activate the plugin: In `mmdetection3d/tools/test.py`, add `import ts_plugin` as the last import statement to activate the plugin.
4. Run the evaluation. 

## Supported Models

- SECOND
- PV-RCNN
- CenterPoint
- Part-A2

## Convert Module Weights
The dimensions of TorchSparse differ from SpConv, so parameter dimension conversion is required. You can use `convert_weights_cmd()` in converter.py as a command line tool or use `convert_weights()` as an API. Both functions have four parameters:

1. `ckpt_before`: Path to the input SpConv checkpoint file.
2. `ckpt_after`: Path where the converted TorchSparse checkpoint will be saved.
3. `cfg_path`: Path to the configuration mmdet3d file of the model.
4. `v_spconv`: Version of SpConv used in the original model (1 or 2).
5. `framework`: Choose between `'openpc'` and `'mmdet3d'`, default to `'mmdet3d'`.  

These parameters allow the converter to locate the input model, specify the output location, understand the model's architecture, and apply the appropriate conversion method based for specific Sparse Conv layers. 

Example conversion commands:
```bash
python examples/converter.py --ckpt_before ../mmdetection3d/models/PV-RCNN/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth --cfg_path ../mmdetection3d/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py --ckpt_after ./converted/PV-RCNN/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth --v_spconv 1 --framework mmdet3d
```


# Run a demo
In your Conda environment, run: 
```bash
python <test_file_path> <cfg_path> <torchsparse_model_path> <cfg_options> --task lidar_det
```

- `test_file_path`: The `tools/test.py` file in mmdet3d repository. 
- `cfg_path`: The path to the mmdet3d's model config for your model. 
- `torchsparse_model_path`: the path to the converted TorchSparse model checkpoint. 
- `cfg_options`: The plugin requires the use of MMDet3D cfg_options to tweak certain model layers to be the plugin layers. `cfg_options` examples are below: 

## SECOND
`cfg_options`:
```bash
"--cfg-options test_evaluator.pklfile_prefix=outputs/torchsparse/second --cfg-options model.middle_encoder.type=SparseEncoderTS"
```

## PV-RCNN
`cfg_options`:
```bash
"--cfg-options test_evaluator.pklfile_prefix=outputs/torchsparse/pv_rcnn --cfg-options model.middle_encoder.type=SparseEncoderTS --cfg-options model.points_encoder.type=VoxelSetAbstractionTS"
```

### CenterPoint Voxel 0.1 Circular NMS

Update the path of the NuScenes dataset in the MMDetection3D dataset config `configs/_base_/datasets/nus-3d.py`.

`cfg_options`:
```bash
"--cfg-options model.pts_middle_encoder.type=SparseEncoderTS"
```