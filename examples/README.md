# Containers
A docker image is created with all the required environment installed: `ioeddk/torchsparse_plugin_demo:latest`, including MMDetection3D, OpenPCDet, TorchSparse, plugins, and PyTorch based on the NVIDIA CUDA 12.1 image.   
The dataset is not included in the image and need to be bind mounted to the container when starting. Specifically with the following command: 
```bash
docker run -it --gpus all --mount type=bind,source=<kitti_dataset_root>,target=/root/data/kitti --mount type=bind,source=<nuscenes_dataset_root>,target=/root/data/nuscenes ioeddk/torchsparse_plugin_demo:latest
```
The above is an example to mount the kitti dataset when starting the container.

Using this container is the simplest way to start the demo of this plugin since the all the dependencies are installed and the paths are configured. You can simply open `/root/repo/torchsparse-dev/examples/mmdetection3d/demo.ipynb` or `/root/repo/torchsparse-dev/examples/openpcdet/demo.ipynb` and run all cells to run the demo. The helper functions in the demo are defined to automatically load the pretrained checkpoints, do the conversions, and run the evaluation. 

If not using the container, then please follow the tutorial below to run the demo. The same copy of demo is also in the demo notebook. 

# Convert the Module Weights
The dimensions of TorchSparse differs from the SpConv, so the parameter dimension conversion is required to use the TorchSparse backend. The conversion script can be found in `examples/converter.py`. The `convert_weighs` function has the header `def convert_weights(ckpt_before: str, ckpt_after: str, cfg_path: str, v_spconv: int = 1, framework: str = "mmdet3d")`:
- `ckpt_before`: the pretrained checkpoint of your module, typically downloaded from the MMDetection3d and OpenPCDet model Zoo. 
- `ckpt_after`: the output path for the converted checkpoint. 
- `cfg_path`: the path to the config file of the MMdet3d or OPC model to be converted. It is requried since the converter create an instance of the model, find all the Sparse Convolution layers, and convert the weights of thay layer. 
- `v_spconv`: the version of the SpConv that the original model is build upon. Valud versions are 1 or 2. 
- `framework`: choose between `mmdet3d` and `openpc`. 

## Example Conversion Commands
### MMDetection3D
```bash
python examples/converter.py --ckpt_before ../mmdetection3d/models/PV-RCNN/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth --cfg_path ../mmdetection3d/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py --ckpt_after ./converted/PV-RCNN/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth --v_spconv 1 --framework mmdet3d
```

### OpenPCDet
```bash
python examples/converter.py --ckpt_before ../OpenPCDet/models/SECOND/second_7862.pth --cfg_path ../OpenPCDet/tools/cfgs/kitti_models/second.yaml --ckpt_after ./converted/SECOND/second_7862.pth --v_spconv 1 --framework openpc
```

# Run evaluation. 
Use the `test.py` that comes with the MMDet3D or OPC to run the evaluation. Provide the converted checkpoint as the model weights. For MMDet3D models, you need to provide extra arguments to replace certain layers to be torchsparse's (see how to replace them in `examples/mmdetection3d/demo.ipynb`). For OpenPCDet, the config file with those layers replaced is in the `examples/openpcdet/cfgs`; to use them, see `examples/openpcdet/demo.ipynb`. An additional step is to add `import ts_plugin` in `mmdetection3d/tools/test.py` and add `import pcdet_plugin` in `OpenPCDet/tools/test.py` to activate the plugins before running the evaluation. 

# Details
Please see `examples/mmdetection3d/demo.ipynb` and `examples/openpcdet/demo.ipynb` for more details. 
