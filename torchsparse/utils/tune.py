import functools
import os
import time
from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, Iterator, List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import torchsparse
from torchsparse import SparseTensor
from torchsparse.nn import Conv3d
from torchsparse.utils import make_ntuple
from torchsparse.nn import functional as F

__all__ = ["tune"]


class StableTimeAccumulator:
    def __init__(self):
        self.fwd_trial = 0
        self.ave_fwd_time = 0.0
        self.bwd_trial = 0
        self.ave_bwd_time = 0.0

    def stable_add(self, cur_fwd_time: float, cur_bwd_time: float):
        if cur_fwd_time > 0:
            if self.fwd_trial == 0:
                self.ave_fwd_time = cur_fwd_time
                self.fwd_trial += 1
            else:
                if cur_fwd_time <= 5 * self.ave_fwd_time:
                    self.ave_fwd_time = (
                        (self.fwd_trial * self.ave_fwd_time) + cur_fwd_time
                    ) / (self.fwd_trial + 1)
                    self.fwd_trial += 1
        if cur_bwd_time > 0:
            if self.bwd_trial == 0:
                self.ave_bwd_time = cur_bwd_time
                self.bwd_trial += 1
            else:
                if cur_bwd_time <= 5 * self.ave_bwd_time:
                    self.ave_bwd_time = (
                        (self.bwd_trial * self.ave_bwd_time) + cur_bwd_time
                    ) / (self.bwd_trial + 1)
                    self.bwd_trial += 1

    def get_total_time(self):
        return self.ave_fwd_time + self.ave_bwd_time


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


def clear_tensor_cache(inputs: SparseTensor):
    inputs = recursive_apply(inputs, lambda x: x._caches.cmaps.clear())
    inputs = recursive_apply(inputs, lambda x: x._caches.kmaps.clear())
    inputs = recursive_apply(inputs, lambda x: x._caches.hashmaps.clear())
    return inputs


def clear_model_config(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, Conv3d):
            module._config = F.conv_config.get_default_conv_config()


def set_group_config(model: nn.Module, names: list, config: Dict):
    for name, module in model.named_modules():
        if isinstance(module, Conv3d):
            if name in names:
                module._config = config.copy()


def torchsparse_tune_timer(
    model: nn.Module,
    inputs: SparseTensor,
    tune_with_bwd: bool,
) -> float:
    fwd_time = 0.0
    bwd_time = 0.0
    torch.cuda.synchronize()
    st = time.time()
    outputs = model(inputs)
    torch.cuda.synchronize()
    ed = time.time()
    fwd_time = ed - st

    if tune_with_bwd:
        top_grad = torch.randn_like(outputs) * 1e-3
        torch.cuda.synchronize()
        st_bp = time.time()
        outputs.backward(top_grad)
        torch.cuda.synchronize()
        ed_bp = time.time()
        bwd_time = ed_bp - st_bp

    return fwd_time, bwd_time


def dataflow_selector(
    model: nn.Module,
    inputs: SparseTensor,
    dataflow_range: List,
    group_to_name: DefaultDict[Tuple[Any, ...], List],
    dataflow_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ],
    tune_with_bwd: bool,
) -> None:

    for group_idx, names in group_to_name.items():
        # Set all configs to default
        clear_model_config(model)
        dummy_config = F.conv_config.get_default_conv_config().copy()

        # Dataflow 1: ImplicitGEMM (Test 2 representative examples)
        if F.Dataflow.ImplicitGEMM in dataflow_range:
            # Setting 1: ImplicitGEMM-unsort
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.ImplicitGEMM,
                False,
                1,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torchsparse_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][(dummy_config.dataflow)].stable_add(
                fwd_duration, bwd_duration
            )

            # Setting 2: ImplicitGEMM-sort(split=3)
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.ImplicitGEMM,
                True,
                3,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torchsparse_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][(dummy_config.dataflow)].stable_add(
                fwd_duration, bwd_duration
            )

        # Dataflow 2: Fetch-On-Demand
        if F.Dataflow.FetchOnDemand in dataflow_range:
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.FetchOnDemand,
                False,
                1,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torchsparse_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][(dummy_config.dataflow)].stable_add(
                fwd_duration, bwd_duration
            )

        # Dataflow 3: Gather-Scatter (Deprecated by default)
        if F.Dataflow.GatherScatter in dataflow_range:
            dummy_config.dataflow, dummy_config.ifsort, dummy_config.split_mask_num = (
                F.Dataflow.GatherScatter,
                False,
                1,
            )
            set_group_config(model, names, dummy_config)
            inputs = clear_tensor_cache(inputs)
            fwd_duration, bwd_duration = torchsparse_tune_timer(
                model, inputs, tune_with_bwd
            )
            dataflow_all[group_idx][(dummy_config.dataflow)].stable_add(
                fwd_duration, bwd_duration
            )


# @torch.no_grad()
def profile_model(
    model: nn.Module,
    inputs: SparseTensor,
    dataflow_range: List,
    dataflow_prune: bool,
    group_to_name: DefaultDict[Tuple[Any, ...], List],
    configs_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ],
    group_dataflow: Dict,
    tune_with_bwd: bool,
) -> None:

    for group_idx, names in group_to_name.items():
        # Set all configs to default
        clear_model_config(model)
        dummy_config = F.conv_config.get_default_conv_config().copy()
        if dataflow_prune:
            local_dataflow_range = [group_dataflow[group_idx]["dataflow"]]
        else:
            local_dataflow_range = dataflow_range

        if F.Dataflow.ImplicitGEMM in local_dataflow_range:
            # Implicit-GEMM. Tune whether to sort & split_mask_num.
            dummy_config.dataflow = F.Dataflow.ImplicitGEMM

            # Stage 1: test unsort fwd
            dummy_config.ifsort = False
            if not tune_with_bwd:
                dummy_config.split_mask_num = 1
                set_group_config(model, names, dummy_config)
                inputs = clear_tensor_cache(inputs)
                fwd_duration, bwd_duration = torchsparse_tune_timer(
                    model, inputs, tune_with_bwd
                )
                configs_all[group_idx][
                    (
                        dummy_config.epsilon,
                        dummy_config.mm_thresh,
                        dummy_config.split_mask_num,
                        dummy_config.split_mask_num_bwd,
                        dummy_config.dataflow,
                        dummy_config.ifsort,
                        dummy_config.FOD_fusion,
                    )
                ].stable_add(fwd_duration, bwd_duration)
            else:
                for split_mask_num_bwd in range(1, 5):
                    dummy_config.split_mask_num_bwd = split_mask_num_bwd
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torchsparse_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    configs_all[group_idx][
                        (
                            dummy_config.epsilon,
                            dummy_config.mm_thresh,
                            dummy_config.split_mask_num,
                            dummy_config.split_mask_num_bwd,
                            dummy_config.dataflow,
                            dummy_config.ifsort,
                            dummy_config.FOD_fusion,
                        )
                    ].stable_add(fwd_duration, bwd_duration)

            # Stage 2: test sort fwd
            dummy_config.ifsort = True
            if not tune_with_bwd:
                for split_mask_num in range(1, 5):
                    dummy_config.split_mask_num = split_mask_num
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torchsparse_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    configs_all[group_idx][
                        (
                            dummy_config.epsilon,
                            dummy_config.mm_thresh,
                            dummy_config.split_mask_num,
                            dummy_config.split_mask_num_bwd,
                            dummy_config.dataflow,
                            dummy_config.ifsort,
                            dummy_config.FOD_fusion,
                        )
                    ].stable_add(fwd_duration, bwd_duration)
            else:
                for split_mask_num in range(1, 5):
                    dummy_config.split_mask_num = split_mask_num
                    dummy_config.split_mask_num_bwd = split_mask_num
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torchsparse_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    for iter in range(1, 5):
                        configs_all[group_idx][
                            (
                                dummy_config.epsilon,
                                dummy_config.mm_thresh,
                                dummy_config.split_mask_num,
                                iter,
                                dummy_config.dataflow,
                                dummy_config.ifsort,
                                dummy_config.FOD_fusion,
                            )
                        ].stable_add(fwd_duration, 0.0)
                        configs_all[group_idx][
                            (
                                dummy_config.epsilon,
                                dummy_config.mm_thresh,
                                iter,
                                dummy_config.split_mask_num_bwd,
                                dummy_config.dataflow,
                                dummy_config.ifsort,
                                dummy_config.FOD_fusion,
                            )
                        ].stable_add(0.0, bwd_duration)

        if F.Dataflow.FetchOnDemand in local_dataflow_range:
            # Fetch-on-Demand. Tune whether to fuse.
            dummy_config.dataflow = F.Dataflow.FetchOnDemand
            for FOD_fusion in [True, False]:
                dummy_config.FOD_fusion = FOD_fusion
                set_group_config(model, names, dummy_config)
                inputs = clear_tensor_cache(inputs)
                fwd_duration, bwd_duration = torchsparse_tune_timer(
                    model, inputs, tune_with_bwd
                )
                configs_all[group_idx][
                    (
                        dummy_config.epsilon,
                        dummy_config.mm_thresh,
                        dummy_config.split_mask_num,
                        dummy_config.split_mask_num_bwd,
                        dummy_config.dataflow,
                        dummy_config.ifsort,
                        dummy_config.FOD_fusion,
                    )
                ].stable_add(fwd_duration, bwd_duration)

        if F.Dataflow.GatherScatter in local_dataflow_range:
            # Gather-Scatter. Tune eps & mm_thresh
            dummy_config.dataflow = F.Dataflow.GatherScatter
            for epsilon in np.arange(0.0, 0.6, 0.1):
                for mm_thresh in [
                    0,
                    5000,
                    7500,
                    10000,
                    12500,
                    15000,
                    17500,
                    20000,
                    22500,
                    25000,
                ]:
                    dummy_config.epsilon, dummy_config.mm_thresh = epsilon, mm_thresh
                    set_group_config(model, names, dummy_config)
                    inputs = clear_tensor_cache(inputs)
                    fwd_duration, bwd_duration = torchsparse_tune_timer(
                        model, inputs, tune_with_bwd
                    )
                    configs_all[group_idx][
                        (
                            dummy_config.epsilon,
                            dummy_config.mm_thresh,
                            dummy_config.split_mask_num,
                            dummy_config.split_mask_num_bwd,
                            dummy_config.dataflow,
                            dummy_config.ifsort,
                            dummy_config.FOD_fusion,
                        )
                    ].stable_add(fwd_duration, bwd_duration)


# @torch.no_grad()
def tune(
    model: nn.Module,
    data_loader: Iterable,
    n_samples: int = 100,
    collect_fn: Callable = lambda data: data,
    enable_fp16: bool = False,
    save_dir: str = ".torchsparse-tune",
    tune_tag: str = "temp",
    force_retune: bool = False,
    dataflow_range: List = [F.Dataflow.ImplicitGEMM],
    dataflow_prune: bool = False,
    tune_with_bwd: bool = False,
    verbose: bool = True,
    skip_warning: bool = False,
):
    """Two-stage tuner for the best configuration by the provided model and data loader.
    Args:
        model: A nn.Module to be profiled for best conv configs.
        data_loader: An iterator with data samples. Recommended
            to use the same data loader for training.
        n_samples: Number of samples for profiling group configs.
        collect_fn: Process data before calling model.forward(). In other words,
            run `model(*collect_fn(data))` where data is yielded by data_loader.
            The default case handles {'input': SparseTensor,...} for data.
    """
    # An iterator can only be used once, so use with care.
    if isinstance(data_loader, Iterator):
        if not skip_warning:
            print(f"Warning: data_loader is an iterator of type {type(data_loader)}.")
            print("Take caution if data_loader is shared with other functions.")
    if not torchsparse.backends.benchmark:  # type: ignore
        if not skip_warning:
            print(
                "Warning: to use tuning, "
                + "torchsparse.backends.benchmark is automatically set to be true."
            )
        torchsparse.backends.benchmark = True  # type: ignore

    dataflow_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ] = defaultdict(lambda: defaultdict(StableTimeAccumulator))
    configs_all: DefaultDict[
        Tuple[Any, ...], DefaultDict[Tuple[Any, ...], StableTimeAccumulator]
    ] = defaultdict(lambda: defaultdict(StableTimeAccumulator))
    name_to_group: DefaultDict[str, Tuple[Any, ...]] = {}
    group_to_name = defaultdict(list)
    device_id = int(str(next(model.parameters()).device).split(":")[-1])

    # hook function to store data for profiling
    # group the conv layers by stage
    def dump(module, inputs, outputs, name):
        if not module.transposed:
            tensor_stride = inputs[0].stride
        else:
            tensor_stride = tuple(
                inputs[0].stride[k] // make_ntuple(module.stride, ndim=3)[k]
                for k in range(3)
            )
        group_idx = (tensor_stride, module.kernel_size, module.stride, module.dilation)
        name_to_group[name] = group_idx
        group_to_name[group_idx].append(name)

    group_dataflow = {}
    group_configs = {}
    if (os.path.exists(os.path.join(save_dir, tune_tag))) and not force_retune:
        if verbose:
            print("Load existing tuned group configs")
        name_to_group, group_configs = torch.load(os.path.join(save_dir, tune_tag))
    else:
        handler_collection = []
        for name, module in model.named_modules():
            # register hook
            if isinstance(module, Conv3d):
                if len(module.kernel.data.shape) == 3:
                    _handler = module.register_forward_hook(
                        functools.partial(dump, name=name)
                    )
                    handler_collection.append(_handler)

        # Stage 0: Dump the model structure
        for i, feed_dict in enumerate(
            tqdm(
                data_loader,
                desc="Dump the model structure",
                leave=False,
                total=n_samples,
            )
        ):
            inputs = collect_fn(feed_dict)
            if enable_fp16:
                inputs = recursive_apply(inputs, lambda x: x.half())
                model = model.half()
            inputs = recursive_apply(inputs, lambda x: x.cuda())
            model = model.cuda()
            with torch.cuda.amp.autocast(enabled=enable_fp16):
                # generate dumps
                name_to_group = {}
                group_to_name = defaultdict(list)
                _ = model(inputs)
                # detach the hook
                for _handler in handler_collection:
                    _handler.remove()
            break

        # Stage 1: select best dataflow for each group (Prune the search space)
        if dataflow_prune:
            if len(dataflow_range) == 1:
                if verbose:
                    print(
                        f"Only 1 dataflow ({dataflow_range[0]}) is set. Skip dataflow selecting."
                    )
                dataflow_prune = False
            else:
                count = 0
                for i, feed_dict in enumerate(
                    tqdm(
                        data_loader,
                        desc="Select best dataflow for each group",
                        leave=False,
                        total=n_samples,
                    )
                ):
                    inputs = collect_fn(feed_dict)
                    if enable_fp16:
                        inputs = recursive_apply(inputs, lambda x: x.half())
                        model = model.half()
                    inputs = recursive_apply(inputs, lambda x: x.cuda())
                    model = model.cuda()
                    with torch.cuda.amp.autocast(enabled=enable_fp16):
                        if i == 0:
                            # device warm-up
                            for warm_iter in range(10):
                                _ = model(inputs)
                        dataflow_selector(
                            model,
                            inputs,
                            dataflow_range,
                            group_to_name,
                            dataflow_all,
                            tune_with_bwd,
                        )
                    count += 1
                    if count == n_samples:
                        break

                # Search for the best dataflow
                for group_idx in dataflow_all:
                    time_min = -1.0
                    for dataflow in dataflow_all[group_idx]:
                        if (
                            time_min < 0
                            or time_min
                            > dataflow_all[group_idx][(dataflow)].get_total_time()
                        ):
                            time_min = dataflow_all[group_idx][
                                (dataflow)
                            ].get_total_time()
                            dataflow_best = dataflow
                    group_dataflow[group_idx] = {"dataflow": dataflow_best}

        # Stage 2: Tune best configs for each group
        count = 0
        for i, feed_dict in enumerate(
            tqdm(
                data_loader,
                desc="Tuning best group configs",
                leave=False,
                total=n_samples,
            )
        ):
            inputs = collect_fn(feed_dict)
            if enable_fp16:
                inputs = recursive_apply(inputs, lambda x: x.half())
                model = model.half()
            inputs = recursive_apply(inputs, lambda x: x.cuda())
            model = model.cuda()
            with torch.cuda.amp.autocast(enabled=enable_fp16):
                if i == 0:
                    # device warm-up
                    for warm_iter in range(10):
                        _ = model(inputs)
                profile_model(
                    model,
                    inputs,
                    dataflow_range,
                    dataflow_prune,
                    group_to_name,
                    configs_all,
                    group_dataflow,
                    tune_with_bwd,
                )
            count += 1
            if count == n_samples:
                break

        # Search for the best configs for each group
        for group_idx in configs_all:
            time_min = -1.0
            for (
                ep,
                thresh,
                split_mask_num,
                split_mask_num_bwd,
                dataflow,
                ifsort,
                FOD_fusion,
            ) in configs_all[group_idx]:
                if (
                    time_min < 0
                    or time_min
                    > configs_all[group_idx][
                        (
                            ep,
                            thresh,
                            split_mask_num,
                            split_mask_num_bwd,
                            dataflow,
                            ifsort,
                            FOD_fusion,
                        )
                    ].get_total_time()
                ):
                    time_min = configs_all[group_idx][
                        (
                            ep,
                            thresh,
                            split_mask_num,
                            split_mask_num_bwd,
                            dataflow,
                            ifsort,
                            FOD_fusion,
                        )
                    ].get_total_time()
                    ep_best = ep
                    thresh_best = thresh
                    split_mask_num_best = split_mask_num
                    split_mask_num_bwd_best = split_mask_num_bwd
                    dataflow_best = dataflow
                    ifsort_best = ifsort
                    FOD_fusion_best = FOD_fusion
            group_configs[group_idx] = {
                "epsilon": ep_best,
                "mm_thresh": thresh_best,
                "split_mask_num": split_mask_num_best,
                "split_mask_num_bwd": split_mask_num_bwd_best,
                "dataflow": dataflow_best,
                "ifsort": ifsort_best,
                "FOD_fusion": FOD_fusion_best,
            }

        # save tuned group configs
        if device_id == 0:
            if verbose:
                print("Save tuned group configs to", os.path.join(save_dir, tune_tag))
            os.makedirs(save_dir, exist_ok=True)
            torch.save((name_to_group, group_configs), os.path.join(save_dir, tune_tag))

    # modify the model
    for name, module in model.named_modules():
        if isinstance(module, Conv3d):
            if name in name_to_group:
                layer_group_idx = name_to_group[name]
                if layer_group_idx in group_configs:
                    new_config = module._config
                    if new_config is None:
                        glb_config = F.conv_config.get_global_conv_config()
                        if glb_config is not None:
                            new_config = glb_config.copy()
                        else:
                            new_config = F.conv_config.get_default_conv_config().copy()
                    new_config.dataflow = group_configs[layer_group_idx]["dataflow"]
                    new_config.epsilon = group_configs[layer_group_idx]["epsilon"]
                    new_config.mm_thresh = group_configs[layer_group_idx]["mm_thresh"]
                    new_config.ifsort = group_configs[layer_group_idx]["ifsort"]
                    new_config.split_mask_num = group_configs[layer_group_idx][
                        "split_mask_num"
                    ]
                    new_config.split_mask_num_bwd = group_configs[layer_group_idx][
                        "split_mask_num_bwd"
                    ]
                    new_config.FOD_fusion = group_configs[layer_group_idx]["FOD_fusion"]
                    module._config = new_config.copy()
