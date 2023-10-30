from typing import List
import os
import torch

__version__ = "2.1.0"


def find_maximal_match(support_list: List, target):
    if target in support_list:
        return target
    else:
        max_match_version = None
        for item in support_list:
            if item <= target:
                max_match_version = item
        if max_match_version == None:
            max_match_version = support_list[0]
            print(
                f"[Warning] CUDA version {target} is too low, may not be well supported by torch_{torch.__version__}."
            )
        return max_match_version


torch_cuda_mapping = dict(
    [
        ("torch19", ["11.1"]),
        ("torch110", ["11.1", "11.3"]),
        ("torch111", ["11.3", "11.5"]),
        ("torch112", ["11.3", "11.6"]),
        ("torch113", ["11.6", "11.7"]),
        ("torch20", ["11.7", "11.8"]),
    ]
)

torch_tag, _ = ("torch" + torch.__version__).rsplit(".", 1)
torch_tag = torch_tag.replace(".", "")

if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    support_cuda_list = torch_cuda_mapping[torch_tag]
    cuda_version = find_maximal_match(support_cuda_list, cuda_version)
    cuda_tag = "cu" + cuda_version
else:
    cuda_tag = "cpu"
cuda_tag = cuda_tag.replace(".", "")


os.system(
    f"pip install --extra-index-url http://24.199.104.228/simple --trusted-host 24.199.104.228 torchsparse=={__version__}+{torch_tag}{cuda_tag} --force-reinstall"
)
