from setuptools import setup, find_packages
from jinja2 import Template
import os

config_template_paths = [
    "./cfgs_templates/kitti_models/second_plugin.yaml",
    "./cfgs_templates/kitti_models/PartA2_plugin.yaml",
    "./cfgs_templates/kitti_models/pv_rcnn_plugin.yaml",
    "./cfgs_templates/kitti_models/voxel_rcnn_car_plugin.yaml",
    "./cfgs_templates/nuscenes_models/cbgs_voxel0075_voxelnext_mini.yaml"
]

os.makedirs("./cfgs", exist_ok=True)
os.makedirs("./cfgs/kitti_models", exist_ok=True)
os.makedirs("./cfgs/nuscenes_models", exist_ok=True)

# define PCDET_BASE
if os.environ.get("PCDET_BASE") is None:
    # throw some exceptions to ask users to deifne the environment variable
    raise ValueError("Please define the environment variable PCDET_BASE")
else:
    base = os.environ.get("PCDET_BASE")
    print(f"PCDET_BASE: {base}")
    for template_path in config_template_paths:
        curr_template = Template(open(template_path).read())
        curr_template_rendered = curr_template.render(pcdet_base_path=base)

        file_name = os.path.basename(template_path)
        folder_path = os.path.dirname(template_path)
        folder_name = os.path.basename(folder_path)
        output_file_path = os.path.join("./cfgs", folder_name, file_name)
        with open(output_file_path, 'w') as file:
            file.write(curr_template_rendered)


    

setup(
    name='pcdet_plugin',
    version='0.1',
    packages=find_packages(),
)

# Define global initialize torchsparse backend
# design init function, let pcdet traverse the folder we modified. Then in pcdet plugin reference folders. 
