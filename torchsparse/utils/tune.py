import functools
import os
import time
from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, Iterator, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import torchsparse
from torchsparse import SparseTensor
from torchsparse.nn import Conv3d
from torchsparse.utils import make_ntuple

__all__ = ['tune']


def recursive_apply(x, func):
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_apply(v, func) for v in x]
    if isinstance(x, tuple):
        return (recursive_apply(v, func) for v in x)
    if isinstance(x, SparseTensor):
        temp = func(x)
        return temp if isinstance(temp, SparseTensor) else x
    return x


@torch.no_grad()
def profile_model(dumps: DefaultDict[str, List],
                  configs_all: DefaultDict[str,
                                           DefaultDict[float,
                                                       DefaultDict[int,
                                                                   float]]],
                  kmap_mode: str, enable_fp16: bool) -> None:
    for name, dump in dumps.items():
        for sample in dump:
            x = sample['inputs']
            p = sample['params']
            dummy_config = {
                'epsilon': 0.1,
                'mm_thresh': 0,
                'kmap_mode': kmap_mode
            }
            layer = Conv3d(p['in_channels'],
                           p['out_channels'],
                           p['kernel_size'],
                           p['stride'],
                           p['dilation'],
                           transposed=p['transposed'],
                           config=dummy_config)
            layer = layer.to(x.F.device).eval().half() \
                if enable_fp16 else layer.to(x.F.device).eval()
            # cache reordered kernel
            layer(x)

            for epsilon in np.arange(0.0, 0.6, 0.1):
                for mm_thresh in [
                        0, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500,
                        25000
                ]:
                    layer.config['epsilon'], layer.config['mm_thresh'] = \
                        epsilon, mm_thresh
                    torch.cuda.synchronize()
                    st = time.time()
                    layer(x)
                    torch.cuda.synchronize()
                    ed = time.time()
                    configs_all[name][epsilon][mm_thresh] += (ed - st)


@torch.no_grad()
def tune(
    model: nn.Module,
    data_loader: Iterable,
    n_samples: int = 100,
    collect_fn: Callable = lambda data: data,
    enable_fp16: bool = False,
    kmap_mode: str = 'hashmap',
    save_dir: str = None,
    tune_id: str = 'temp',
):
    """Search for the best group strategy by the provided model and data loader.

    Args:
        model: A nn.Module to be profiled for best group configs.
        data_loader: An iterator with data samples. Recommended
            to use the same data loader for training.
        n_samples: Number of samples for profiling group configs.
        collect_fn: Process data before calling model.forward(). In other words,
            run `model(*collect_fn(data))` where data is yielded by data_loader.
            The default case handles {'input': SparseTensor,...} for data.
    """
    # An iterator can only be used once, so use with care.
    if isinstance(data_loader, Iterator):
        print(
            f'Warning: data_loader is an iterator of type {type(data_loader)}.')
        print('Take caution if data_loader is shared with other functions.')
    if not torchsparse.backends.benchmark:  # type: ignore
        print(
            'Warning: to use tuning, '
            + 'torchsparse.backends.benchmark is automatically set to be true.')
        torchsparse.backends.benchmark = True  # type: ignore
    configs_all: DefaultDict[str, DefaultDict[float, DefaultDict[int, float]]] \
        = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    dumps = defaultdict(list)
    device_id = int(str(next(model.parameters()).device).split(':')[-1])

    # hook function to store data for profiling
    def dump(module, inputs, outputs, name):
        if not module.transposed:
            kmap = inputs[0].kmaps.get(
                (inputs[0].stride, make_ntuple(module.kernel_size, ndim=3),
                 make_ntuple(module.stride,
                             ndim=3), make_ntuple(module.dilation, ndim=3)))
        else:
            tensor_stride = tuple(inputs[0].stride[k]
                                  // make_ntuple(module.stride, ndim=3)[k]
                                  for k in range(3))
            kmap = inputs[0].kmaps[(tensor_stride,
                                    make_ntuple(module.kernel_size, ndim=3),
                                    make_ntuple(module.stride, ndim=3),
                                    make_ntuple(module.dilation, ndim=3))]
        dumps[name].append({
            'inputs': inputs[0],
            'neighbor_offset': kmap[1].tolist() if kmap is not None else None,
            'params': {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'dilation': module.dilation,
                'transposed': module.transposed
            }
        })

    group_configs = {}
    if save_dir is None:
        save_dir = os.path.expanduser('~/.torchsparse')
    if os.path.exists(os.path.join(save_dir, tune_id)):
        print('Load existing tuned group configs')
        group_configs = torch.load(os.path.join(save_dir, tune_id))
    else:
        handler_collection = []
        for name, module in model.named_modules():
            # register hook
            if isinstance(module, Conv3d):
                if (len(module.kernel.data.shape) == 3):
                    _handler = module.register_forward_hook(
                        functools.partial(dump, name=name))
                    handler_collection.append(_handler)
        model = model.eval()

        count = 0
        for i, feed_dict in enumerate(
                tqdm(data_loader,
                     desc='Tuning best group configs',
                     leave=False,
                     total=n_samples)):
            inputs = collect_fn(feed_dict)
            if enable_fp16:
                inputs = recursive_apply(inputs, lambda x: x.half())
                model = model.half()
            inputs = recursive_apply(inputs, lambda x: x.cuda())
            model = model.cuda()
            with torch.cuda.amp.autocast(enabled=enable_fp16):
                # doing the warm-up
                if i == 0:
                    for _ in range(10):
                        _ = model(inputs)
                        inputs = recursive_apply(inputs,
                                                 lambda x: x.cmaps.clear())
                        inputs = recursive_apply(inputs,
                                                 lambda x: x.kmaps.clear())
                        dumps = defaultdict(list)
                model(inputs)
                profile_model(dumps, configs_all, kmap_mode, enable_fp16)
            dumps = defaultdict(list)
            count += 1
            if count == n_samples:
                break
        for _handler in handler_collection:
            _handler.remove()

        for name in configs_all:
            time_layer_min = 0.0
            for ep in configs_all[name]:
                for thresh in configs_all[name][ep]:
                    if time_layer_min == 0 or time_layer_min > configs_all[
                            name][ep][thresh]:
                        time_layer_min = configs_all[name][ep][thresh]
                        ep_best = ep
                        thresh_best = thresh
            group_configs[name] = {'epsilon': ep_best, 'mm_thresh': thresh_best}
        # save tuned group configs
        if device_id == 0:
            print('Save tuned group configs to',
                  os.path.join(save_dir, tune_id))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(group_configs, os.path.join(save_dir, tune_id))

    for name, module in model.named_modules():
        if isinstance(module, Conv3d):
            if name in group_configs:
                module.config['epsilon'] = group_configs[name]['epsilon']
                module.config['mm_thresh'] = group_configs[name]['mm_thresh']
