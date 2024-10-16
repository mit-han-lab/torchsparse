import torch
import re
import numpy as np

conv_weights = [
    "backbone_3d.conv_input.0.weight",
    "backbone_3d.conv1.0.conv1.weight",
    "backbone_3d.conv1.0.conv2.weight",
    "backbone_3d.conv1.1.conv1.weight",
    "backbone_3d.conv1.1.conv2.weight",
    "backbone_3d.conv2.0.0.weight",
    "backbone_3d.conv2.1.conv1.weight",
    "backbone_3d.conv2.1.conv2.weight",
    "backbone_3d.conv2.2.conv1.weight",
    "backbone_3d.conv2.2.conv2.weight",
    "backbone_3d.conv3.0.0.weight",
    "backbone_3d.conv3.1.conv1.weight",
    "backbone_3d.conv3.1.conv2.weight",
    "backbone_3d.conv3.2.conv1.weight",
    "backbone_3d.conv3.2.conv2.weight",
    "backbone_3d.conv4.0.0.weight",
    "backbone_3d.conv4.1.conv1.weight",
    "backbone_3d.conv4.1.conv2.weight",
    "backbone_3d.conv4.2.conv1.weight",
    "backbone_3d.conv4.2.conv2.weight",
    "backbone_3d.conv5.0.0.weight",
    "backbone_3d.conv5.1.conv1.weight",
    "backbone_3d.conv5.1.conv2.weight",
    "backbone_3d.conv5.2.conv1.weight",
    "backbone_3d.conv5.2.conv2.weight",
    "backbone_3d.conv6.0.0.weight",
    "backbone_3d.conv6.1.conv1.weight",
    "backbone_3d.conv6.1.conv2.weight",
    "backbone_3d.conv6.2.conv1.weight",
    "backbone_3d.conv6.2.conv2.weight",
    # "backbone_3d.conv_out.0.weight",
    "backbone_3d.shared_conv.0.weight",
    "dense_head.heads_list.0.center.0.0.weight",
    "dense_head.heads_list.0.center.1.weight",
    "dense_head.heads_list.0.center_z.0.0.weight",
    "dense_head.heads_list.0.center_z.1.weight",
    "dense_head.heads_list.0.dim.0.0.weight",
    "dense_head.heads_list.0.dim.1.weight",
    "dense_head.heads_list.0.rot.0.0.weight",
    "dense_head.heads_list.0.rot.1.weight",
    "dense_head.heads_list.0.hm.0.0.weight",
    "dense_head.heads_list.0.hm.1.weight",
    "dense_head.heads_list.1.center.0.0.weight",
    "dense_head.heads_list.1.center.1.weight",
    "dense_head.heads_list.1.center_z.0.0.weight",
    "dense_head.heads_list.1.center_z.1.weight",
    "dense_head.heads_list.1.dim.0.0.weight",
    "dense_head.heads_list.1.dim.1.weight",
    "dense_head.heads_list.1.rot.0.0.weight",
    "dense_head.heads_list.1.rot.1.weight",
    "dense_head.heads_list.1.hm.0.0.weight",
    "dense_head.heads_list.1.hm.1.weight",
    "dense_head.heads_list.2.center.0.0.weight",
    "dense_head.heads_list.2.center.1.weight",
    "dense_head.heads_list.2.center_z.0.0.weight",
    "dense_head.heads_list.2.center_z.1.weight",
    "dense_head.heads_list.2.dim.0.0.weight",
    "dense_head.heads_list.2.dim.1.weight",
    "dense_head.heads_list.2.rot.0.0.weight",
    "dense_head.heads_list.2.rot.1.weight",
    "dense_head.heads_list.2.hm.0.0.weight",
    "dense_head.heads_list.2.hm.1.weight",
    "dense_head.heads_list.3.center.0.0.weight",
    "dense_head.heads_list.3.center.1.weight",
    "dense_head.heads_list.3.center_z.0.0.weight",
    "dense_head.heads_list.3.center_z.1.weight",
    "dense_head.heads_list.3.dim.0.0.weight",
    "dense_head.heads_list.3.dim.1.weight",
    "dense_head.heads_list.3.rot.0.0.weight",
    "dense_head.heads_list.3.rot.1.weight",
    "dense_head.heads_list.3.hm.0.0.weight",
    "dense_head.heads_list.3.hm.1.weight",
    "dense_head.heads_list.4.center.0.0.weight",
    "dense_head.heads_list.4.center.1.weight",
    "dense_head.heads_list.4.center_z.0.0.weight",
    "dense_head.heads_list.4.center_z.1.weight",
    "dense_head.heads_list.4.dim.0.0.weight",
    "dense_head.heads_list.4.dim.1.weight",
    "dense_head.heads_list.4.rot.0.0.weight",
    "dense_head.heads_list.4.rot.1.weight",
    "dense_head.heads_list.4.hm.0.0.weight",
    "dense_head.heads_list.4.hm.1.weight",
    "dense_head.heads_list.5.center.0.0.weight",
    "dense_head.heads_list.5.center.1.weight",
    "dense_head.heads_list.5.center_z.0.0.weight",
    "dense_head.heads_list.5.center_z.1.weight",
    "dense_head.heads_list.5.dim.0.0.weight",
    "dense_head.heads_list.5.dim.1.weight",
    "dense_head.heads_list.5.rot.0.0.weight",
    "dense_head.heads_list.5.rot.1.weight",
    "dense_head.heads_list.5.hm.0.0.weight",
    "dense_head.heads_list.5.hm.1.weight",
    "dense_head.heads_list.0.vel.0.0.weight",
    "dense_head.heads_list.0.vel.1.weight",
    "dense_head.heads_list.1.vel.0.0.weight",
    "dense_head.heads_list.1.vel.1.weight",
    "dense_head.heads_list.2.vel.0.0.weight",
    "dense_head.heads_list.2.vel.1.weight",
    "dense_head.heads_list.3.vel.0.0.weight",
    "dense_head.heads_list.3.vel.1.weight",
    "dense_head.heads_list.4.vel.0.0.weight",
    "dense_head.heads_list.4.vel.1.weight",
    "dense_head.heads_list.5.vel.0.0.weight",
    "dense_head.heads_list.5.vel.1.weight"
]

no_squeeze = [
    "dense_head.heads_list.0.center_z.1.weight",
	"dense_head.heads_list.0.hm.1.weight",
	"dense_head.heads_list.1.center_z.1.weight",
	"dense_head.heads_list.2.center_z.1.weight",
	"dense_head.heads_list.3.center_z.1.weight",
	"dense_head.heads_list.3.hm.1.weight",
	"dense_head.heads_list.4.center_z.1.weight",
	"dense_head.heads_list.5.center_z.1.weight"
]


def convert_weights_2d(key, model):
    new_key = key.replace(".weight", ".kernel")
    weights = model[key]
    oc, kx, ky, ic = weights.shape
    
    converted_weights = weights.reshape(oc, -1, ic)

    converted_weights = converted_weights.permute(1, 0, 2)

    # find order of dimension
    # weight_order = [] 
    # for weight in converted_weights:
    #     weight_order.append(torch.sum(weight))

    # weight_order_rearranged = []

    if converted_weights.shape[0] == 1:
        converted_weights = converted_weights[0]
        converted_weights = converted_weights.permute(1,0)
    elif converted_weights.shape[0] == 9:
        offsets = [ list(range(ky)), list(range(kx))]
        offsets = [
            (x * ky + y)
            for y in offsets[0]
            for x in offsets[1]
        ]
        offsets = torch.tensor(
            offsets, dtype=torch.int64, device=converted_weights.device
        )
        converted_weights = converted_weights[offsets]
        converted_weights = converted_weights.permute(0,2,1)
        # for weight in converted_weights:
        #     weight_order_rearranged.append(torch.sum(weight))
    
    return new_key, converted_weights

def convert_unit_weight(key, model):
    new_key = key.replace(".weight", ".kernel")
    weight = model[key]
    oc, kx, ky, ic = weight.shape
    new_weight = weight.transpose(0,1).reshape(1,ic,oc)
    # if key in no_squeeze:
    #     return new_key, new_weight
    # else:
    #     return new_key, torch.squeeze(new_weight)
    return new_key, torch.squeeze(new_weight, [0])


def convert_weights_3d(key, model):
    new_key = key.replace(".weight", ".kernel")
    weights = model[key]
    oc, kx, ky, kz, ic = weights.shape
    
    converted_weights = weights.reshape(oc, -1, ic)

    # [oc, x*y*z, ic] -> [x*y*z, oc, ic]
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

    # [x*y*z, oc, ic] -> [x*y*z, ic, oc]
    converted_weights = converted_weights.permute(0,2,1)
    
    return new_key, converted_weights

def convert_voxelnext(ckpt_before, ckpt_after):
    cp_old = torch.load(ckpt_before, map_location="cpu")
    try:
        model = cp_old['model_state']
    except:
        model = cp_old
    new_model = dict()

    for key in model:
        is_sparseconv_weight = False
        if key in conv_weights: # and not re.search(r'lateral_layer', key)
            # is_sparseconv_weight = len(model[key].shape) > 1  # dimension larger than 1
            is_sparseconv_weight = True

        if is_sparseconv_weight:
            if len(model[key].shape) == 5:
                new_key, converted_weights = convert_weights_3d(key, model)
                if key == 'backbone_3d.conv_input.0.weight':
                    # converted_weights = converted_weights[:, :-1, :]
                    converted_weights = converted_weights[:, :, :]
            elif np.prod(model[key].shape[1:-1]) == 1:  # kernel size is 1
                # This is a 2d kernel with the kerne size of 1. 
                new_key, converted_weights = convert_unit_weight(key, model)
            elif len(model[key].shape) == 4:
                new_key, converted_weights = convert_weights_2d(key, model)
            else:
                new_key = key.replace(".weight", ".kernel")
                converted_weights = model[key]
        else:
            new_key = key
            converted_weights = model[key]

        new_model[new_key] = converted_weights

    cp_old['model_state'] = new_model
    torch.save(cp_old, ckpt_after)


convert_voxelnext("/home/yingqi/repo/OpenPCDet/models/VoxelNeXt/voxelnext_nuscenes_kernel1.pth", "/home/yingqi/repo/torchsparse-dev/converted_models/openpcdet/VoxelNeXt/voxelnext_nuscenes_kernel1.pth")