import torch

__all__ = ['global_avg_pool', 'global_max_pool']


def global_avg_pool(inputs):
    batch_index = inputs.C[:, -1]
    max_index = torch.max(batch_index).item()
    outputs = []
    for i in range(max_index + 1):
        cur_inputs = torch.index_select(inputs.F, 0,
                                        torch.where(batch_index == i)[0])
        cur_outputs = cur_inputs.mean(0).unsqueeze(0)
        outputs.append(cur_outputs)
    outputs = torch.cat(outputs, 0)
    return outputs


def global_max_pool(inputs):
    batch_index = inputs.C[:, -1]
    max_index = torch.max(batch_index).item()
    outputs = []
    for i in range(max_index + 1):
        cur_inputs = torch.index_select(inputs.F, 0,
                                        torch.where(batch_index == i)[0])
        cur_outputs = cur_inputs.max(0)[0].unsqueeze(0)
        outputs.append(cur_outputs)
    outputs = torch.cat(outputs, 0)
    return outputs
