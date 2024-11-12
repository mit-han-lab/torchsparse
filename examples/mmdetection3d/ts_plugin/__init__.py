from .models import *

from mmengine.registry import MODELS

from torchsparse.nn import BatchNorm

MODELS.register_module('TorchSparseBatchNorm', force=True, module=BatchNorm)