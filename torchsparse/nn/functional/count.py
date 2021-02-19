import torchsparse_backend

__all__ = ['count']


def count(idx, num):
    idx = idx.contiguous()
    if idx.device.type == 'cuda':
        return torchsparse_backend.count_forward_cuda(idx, num)
    elif idx.device.type in ['cpu', 'tpu']:
        return torchsparse_backend.count_forward_cpu(idx, num)
    else:
        raise NotImplementedError
