from .models import *

# Apply Monkey Patch
# Monkey Patch here
import pcdet.models.backbones_3d as pcd_backbones_3d
import pcdet.models.backbones_3d.pfe as pcd_pfe
import pcdet.models.backbones_2d.map_to_bev as pcd_map_to_bev
import pcdet.models.dense_heads as pcd_dense_heads
import pcdet.models.roi_heads as pcd_roi_heads

import pcdet_plugin.models.backbones_3d as pcd_plugin_backbones_3d
import pcdet_plugin.models.backbones_3d.pfe as pcd_plugin_pfe
import pcdet_plugin.models.backbones_3d.unet as pcd_plugin_spconv_unet
import pcdet_plugin.models.backbones_3d.backbone_voxel_next as pcd_plugin_backbone_voxel_next
import pcdet_plugin.models.backbones_2d.map_to_bev as pcd_plugin_map_to_bev
import pcdet_plugin.models.dense_heads.voxel_next_head as pcd_plugin_voxel_next_head
import pcdet_plugin.models.roi_heads.partA2_head as pcd_plugin_partA2_head

import pcdet_plugin.models.detectors.detector3d_template as pcd_plugin_detector3d_template

pcd_backbones_3d.__all__['VoxelBackBone8xTS'] = pcd_plugin_backbones_3d.VoxelBackBone8xTS
pcd_backbones_3d.__all__['UNetV2TS'] = pcd_plugin_spconv_unet.UNetV2TS
pcd_backbones_3d.__all__['VoxelResBackBone8xVoxelNeXtTS'] = pcd_plugin_backbone_voxel_next.VoxelResBackBone8xVoxelNeXtTS
pcd_map_to_bev.__all__['HeightCompressionTS'] = pcd_plugin_map_to_bev.HeightCompressionTS
pcd_pfe.__all__['VoxelSetAbstractionTS'] = pcd_plugin_pfe.VoxelSetAbstractionTS
pcd_dense_heads.__all__['VoxelNeXtHeadTS'] = pcd_plugin_voxel_next_head.VoxelNeXtHeadTS
pcd_roi_heads.__all__['PartA2FCHeadTS'] = pcd_plugin_partA2_head.PartA2FCHeadTS

# Monkey patch the detector 3d template
import pcdet.models.detectors as pcd_detectors

pcd_detectors.__all__['Detector3DTemplate']._load_state_dict = pcd_plugin_detector3d_template.Detector3DTemplate._load_state_dict
# pcd_detectors.detector3d_template.Detector3DTemplate = pcd_plugin_detector3d_template.Detector3DTemplate