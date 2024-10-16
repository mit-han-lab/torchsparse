"""This is the model converter to convert a SpConv model to TorchSparse model. 
"""
import argparse
import torch
import re
import logging
import spconv.pytorch as spconv
import logging

# Disable JIT because running OpenPCDet with JIT enabled will cause some import issue. 
torch.jit._state.disable()

# Works for SECOND
def convert_weights_v2(key, model):
    """Convert model weights for models build with SpConv v2. 

    :param key: _description_
    :type key: _type_
    :param model: _description_
    :type model: _type_
    :return: _description_
    :rtype: _type_
    """
    new_key = key.replace(".weight", ".kernel")
    weights = model[key]
    oc, kx, ky, kz, ic = weights.shape
    
    converted_weights = weights.reshape(oc, -1, ic)

    converted_weights = converted_weights.permute(1, 0, 2)

    if converted_weights.shape[0] == 1:
        converted_weights = converted_weights[0]
    elif converted_weights.shape[0] == 27:
        offsets = [list(range(kz)), list(range(ky)), list(range(kx))]
        kykx = ky * kx
        offsets = [
            (x * kykx + y * kx + z)
            for z in offsets[0]
            for y in offsets[1]
            for x in offsets[2]
        ]
        offsets = torch.tensor(
            offsets, dtype=torch.int64, device=converted_weights.device
        )
        converted_weights = converted_weights[offsets]

    converted_weights = converted_weights.permute(0,2,1)
    
    return new_key, converted_weights

# Order for CenterPoint, PV-RCNN, and default, legacy SpConv
def convert_weights_v1(key, model):
    """Convert model weights for models implemented with SpConv v1

    :param key: _description_
    :type key: _type_
    :param model: _description_
    :type model: _type_
    :return: _description_
    :rtype: _type_
    """
    new_key = key.replace(".weight", ".kernel")
    weights = model[key]

    kx, ky, kz, ic, oc = weights.shape

    converted_weights = weights.reshape(-1, ic, oc)
    if converted_weights.shape[0] == 1:
        converted_weights = converted_weights[0]

    elif converted_weights.shape[0] == 27:  
        offsets = [list(range(kz)), list(range(ky)), list(range(kx))]
        kykx = ky * kx
        offsets = [
            (x * kykx + y * kx + z)
            for z in offsets[0]
            for y in offsets[1]
            for x in offsets[2]
        ]
        offsets = torch.tensor(
            offsets, dtype=torch.int64, device=converted_weights.device
        )
        converted_weights = converted_weights[offsets]
    elif converted_weights.shape[0] == 3:  # 3 is the case in PartA2. 
        pass
        # offsets = torch.tensor(
        #     [2, 1, 0], dtype=torch.int64, device=converted_weights.device
        # )
        # converted_weights = converted_weights[offsets]
    return new_key, converted_weights

def build_mmdet_model_from_cfg(cfg_path, ckpt_path):
    try:
        from mmdet3d.apis import init_model
        from mmengine.config import Config
    except:
        print("MMDetection3D is not installed. Please install MMDetection3D to use this function.")
    cfg = Config.fromfile(cfg_path)
    model = init_model(cfg, ckpt_path)
    return model

def build_opc_model_from_cfg(cfg_path):
    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network
    except Exception as e:
        print(e)
        raise ImportError("Failed to import OpenPCDet")
    cfg_from_yaml_file(cfg_path, cfg)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        training=False,
        logger=logging.Logger("Build Dataloader"),
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    return model

# Allow use the API to convert based on a passed in model. 
def convert_model_weights(ckpt_before, ckpt_after, model, legacy=False):

    model_modules = {}
    for key, value in model.named_modules():
        model_modules[key] = value

    cp_old = torch.load(ckpt_before, map_location="cpu")
    try:
        opc = False
        old_state_dict = cp_old["state_dict"]
    except:
        opc = True        
        old_state_dict = cp_old["model_state"]

    new_model = dict()

    for state_dict_key in old_state_dict.keys():
        is_sparseconv_weight = False
        if state_dict_key.endswith(".weight"):
            if state_dict_key[:-len(".weight")] in model_modules.keys():
                if isinstance(model_modules[state_dict_key[:-len(".weight")]], (spconv.SparseConv3d, spconv.SubMConv3d, spconv.SparseInverseConv3d)):
                    is_sparseconv_weight = True
            
        if is_sparseconv_weight:
            # print(f"{state_dict_key} is a sparseconv weight")
            pass
        
        if is_sparseconv_weight:
            if len(old_state_dict[state_dict_key].shape) == 5:
                if legacy:
                    new_key, converted_weights = convert_weights_v1(state_dict_key, old_state_dict)
                else:
                    new_key, converted_weights = convert_weights_v2(state_dict_key, old_state_dict)
        else:
            new_key = state_dict_key
            converted_weights = old_state_dict[state_dict_key]

        new_model[new_key] = converted_weights

    if opc:
        cp_old["model_state"] = new_model
    else:
        cp_old["state_dict"] = new_model
    torch.save(cp_old, ckpt_after)


def convert_weights_cmd():
    """Convert the weights of a model from SpConv to TorchSparse.

    :param ckpt_before: Path to the SpConv checkpoint
    :type ckpt_before: str
    :param ckpt_after: Path to the output folder of the converted checkpoint. 
    :type ckpt_after: str
    :param v_spconv: SpConv version used for the weights. Can be one of 1 or 2, defaults to "1"
    :type v_spconv: str, optional
    :param framework: From which framework does the model weight comes from, choose one of mmdet3d or openpc, defaults to "mmdet3d"
    :type framework: str, optional
    """
    # ckpt_before, ckpt_after, v_spconv="1", framework="mmdet3d"

    # argument parser
    parser = argparse.ArgumentParser(description="Convert SpConv model to TorchSparse model")
    parser.add_argument("--ckpt_before", help="Path to the SpConv checkpoint")
    parser.add_argument("--ckpt_after", help="Path to the output folder of the converted checkpoint.")
    parser.add_argument("--cfg_path", help="Path to the config file of the model")
    parser.add_argument("--v_spconv", default="1", help="SpConv version used for the weights. Can be one of 1 or 2")
    parser.add_argument("--framework", default="mmdet3d", help="From which framework does the model weight comes from, choose one of mmdet3d or openpc")
    args = parser.parse_args()

    # Check the plugin argument
    assert args.framework in ['mmdet3d', 'openpc'], "plugin argument can only be mmdet3d or openpcdet"
    assert args.v_spconv in ['1', '2'], "v_spconv argument can only be 1 or 2"

    legacy = True if args.v_spconv == "1" else False
    cfg_path = args.cfg_path

    model = build_mmdet_model_from_cfg(cfg_path, args.ckpt_before) if args.framework == "mmdet3d" else build_opc_model_from_cfg(cfg_path)
    convert_model_weights(
        ckpt_before=args.ckpt_before,
        ckpt_after=args.ckpt_after,
        model=model,
        legacy=legacy)
    

def convert_weights(ckpt_before: str, ckpt_after: str, cfg_path: str, v_spconv: int = 1, framework: str = "mmdet3d"):
    """Convert the weights of a model from SpConv to TorchSparse.

    :param ckpt_before: _description_
    :type ckpt_before: str
    :param ckpt_after: _description_
    :type ckpt_after: str
    :param cfg_path: _description_
    :type cfg_path: str
    :param v_spconv: _description_, defaults to 1
    :type v_spconv: int, optional
    :param framework: _description_, defaults to "mmdet3d"
    :type framework: str, optional
    """

    # Check the plugin argument
    assert framework in ['mmdet3d', 'openpc'], "plugin argument can only be mmdet3d or openpcdet"
    assert v_spconv in [1, 2], "v_spconv argument can only be 1 or 2"

    legacy = True if v_spconv == 1 else False

    model = build_mmdet_model_from_cfg(cfg_path, ckpt_before) if framework == "mmdet3d" else build_opc_model_from_cfg(cfg_path)
    convert_model_weights(
        ckpt_before=ckpt_before,
        ckpt_after=ckpt_after,
        model=model,
        legacy=legacy)


if __name__ == "__main__":
    convert_weights_cmd()
    print("Conversion completed")
