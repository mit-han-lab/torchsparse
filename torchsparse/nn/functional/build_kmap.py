from typing import Tuple, Union

import torch

import torchsparse.backend
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['build_kernel_map']


def build_kernel_map(_coords: torch.Tensor,
                     kernel_size: Union[int, Tuple[int, ...]] = 2,
                     stride: Union[int, Tuple[int, ...]] = 2,
                     tensor_stride: Union[int, Tuple[int, ...]] = 1,
                     mode='hashmap') -> torch.Tensor:
    if mode == 'grid':
        coords = _coords[:, [3, 0, 1, 2]]
        stride = make_ntuple(stride, ndim=3)
        kernel_size = make_ntuple(kernel_size, ndim=3)
        tensor_stride = make_ntuple(tensor_stride, ndim=3)
        subm = not (any(s > 1 for s in stride))
        stride = torch.tensor(stride, dtype=torch.int, device=coords.device)
        kernel_size = torch.tensor(kernel_size,
                                   dtype=torch.int,
                                   device=coords.device)
        tensor_stride = torch.tensor(tensor_stride,
                                     dtype=torch.int,
                                     device=coords.device)

        if subm:
            func = torchsparse.backend.build_kernel_map_subm
        else:
            func = torchsparse.backend.build_kernel_map_downsample
        out = func(coords,
                   coords.min(0).values,
                   coords.max(0).values, kernel_size, stride, tensor_stride)

        nbmaps = out[0]
        input_mask, output_mask = out[-2:]
        if len(out) == 4:
            return out
        else:
            return tuple(out[:2]) + (out[2][:, [1, 2, 3, 0]],) + tuple(out[3:])
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     stride=tensor_stride,
                                     device=_coords.device)

        references = F.sphash(_coords)
        kernel_size = make_ntuple(kernel_size, ndim=3)
        stride = make_ntuple(stride, ndim=3)
        if any(s > 1 for s in stride):
            coords = F.spdownsample(_coords, stride, kernel_size, tensor_stride)
        else:
            coords = _coords
        queries = F.sphash(coords, offsets)
        results = F.sphashquery(queries, references)
        nbsizes = torch.sum(results != -1, dim=1)
        nbmaps = torch.nonzero(results != -1)
        nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1)
                                        + nbmaps[:, 1]]
        # important for build masks
        nbmaps = nbmaps.contiguous()
        input_mask, output_mask = torchsparse.backend.build_mask_from_kmap(
            _coords.shape[0], coords.shape[0], nbmaps.int(), nbsizes.int())

        if any(s > 1 for s in stride):
            return nbmaps, nbsizes, coords, input_mask, output_mask
        else:
            return nbmaps, nbsizes, input_mask, output_mask
