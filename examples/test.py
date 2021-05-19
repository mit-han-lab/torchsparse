import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import numpy as np

#inputs = torchsparse.SparseTensor(, torch.Tensor().int())
feat = np.array([[0],[1],[2],[3],[4],[5]], dtype=np.float32)
pos = np.array([[0,3,1,0],[1,1,1,0],[2,2,1,0],[1,4,1,0],[3,0,1,0],[4,3,1,0]], dtype=np.int32)
kernel_size = 3
kernel = torch.ones(kernel_size**3, 1,1)
#kernel[13,0,0] = 1

# pos = np.array([[1,1,1,0],[0,0,0,0]]).astype(np.int32) # last dim is batch dim
# feat = np.array([[1.],[2.]], dtype=np.float32)
label = torch.zeros(len(feat))

inds, labels, inv = torchsparse.utils.sparse_quantize(pos, feat, label, return_index=True, return_invs=True)
#print(kernel)

inputs = torchsparse.SparseTensor(torch.from_numpy(feat[inds]), torch.from_numpy(pos[inds]))
print(inputs.C)
print(inputs.F)

cuda = True
if cuda:
    inputs = inputs.to('cuda')
    kernel = kernel.to('cuda')

print('with autocast:')
#with torch.cuda.amp.autocast():
inputs.F = inputs.F.half()
kernel = kernel.half()
out = torchsparse.nn.functional.conv3d(inputs, kernel, kernel_size=kernel_size, stride=1).to('cpu')
print(out.C)
print(out.F)
print(out.F.dtype)
print(out.F.shape)

# print(inputs.F.dtype)
# print('without autocast:')
# out = torchsparse.nn.functional.conv3d(inputs, kernel, kernel_size=kernel_size, stride=1).to('cpu')
# print(out.C)
# print(out.F)
# print(out.F.dtype) 