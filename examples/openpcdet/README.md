# TorchSparse for OpenPCDet Plugin Demo

This tutorial demonstrates how to evaluate TorchSparse integrated OpenPCDet models. Follow the steps below to install dependencies, configure paths, convert model weights, and run the demo.

## Dependencies

1. **Conda**: Ensure Conda is installed.
2. **OpenPCDet Installation**: Follow the [OpenPCDet documentation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).
3. **Dataset Preparation**: Pre-process the datasets as described [here](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md).
4. **TorchSparse Installation**: Install [TorchSparse](https://github.com/mit-han-lab/torchsparse).
5. **Install TorchSparse Plugin for OpenPCDet**:
    1. Clone this repository.
    2. Define the environment variable `PCDET_BASE` to point to the installation path of OpenPCDet.
    3. Navigate to `examples/openpcdet` and run `pip install -v -e .`.

## Notes

- You may need to disable PyTorch JIT compile to avoid errors. Add the following to the import section of the relevant `.py` file:
  ```python
  import torch
  torch.jit._state.disable()
  ```
- Modify dataset paths in the model config to absolute paths to avoid `FileNotFoundError`.

## Steps

1. Install the dependencies.
2. Specify the base paths and model registry.
3. **IMPORTANT,** Activate the plugin: In `OpenPCDet/tools/test.py`, add `import pcdet_plugin` as the last import statement to activate the plugin.
4. Run the evaluation. 

## Supported Models

- Kitti: SECOND, PV-RCNN, Part-A2
- NuScenes: VoxelNeXt

## Load the Weight Conversion Module
The dimensions of TorchSparse differ from SpConv, so parameter dimension conversion is required. You can use `convert_weights_cmd()` in converter.py as a command line tool or use `convert_weights()` as an API. Both functions have four parameters:

1. `ckpt_before`: Path to the input SpConv checkpoint file.
2. `ckpt_after`: Path where the converted TorchSparse checkpoint will be saved.
3. `cfg_path`: Path to the configuration mmdet3d file of the model.
4. `v_spconv`: Version of SpConv used in the original model (1 or 2).
5. `framework`: Choose between `'openpc'` and `'mmdet3d'`, default to `'mmdet3d'`.  

These parameters allow the converter to locate the input model, specify the output location, understand the model's architecture, and apply the appropriate conversion method based for specific Sparse Conv layers.

Example conversion commands:
```bash
python examples/converter.py --ckpt_before ../OpenPCDet/models/SECOND/second_7862.pth --cfg_path ../OpenPCDet/tools/cfgs/kitti_models/second.yaml --ckpt_after ./converted/SECOND/second_7862.pth --v_spconv 1 --framework openpc
```


## Run the Evaluation
In your Conda environment with all the dependencies installed, run the following for the evaluation: 
```bash
python <test_file_path> --cfg_file <torchsparse_cfg_path> --ckpt <torchsparse_model_path>
```

- `test_file_path`: the evaluatino script in OpenPC. 
- `torchsparse_cfg_path`: the config file of the model, in `examples/openpcdet/cfgs` folder of this repository. 
- `torchsparse_model_path`: converted TorchSparse checkpoint. 


### VoxelNeXt
VoxelNeXt requires `examples/openpcdet/converter_voxelnext.py` as a model converter, rather than the general converter.py.
