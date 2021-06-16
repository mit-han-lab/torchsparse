import os
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
import tvm
from tvm import te

from utils import load_tensor
from mm_test import matmul_d2l, matmul_topi, bench_matmul_tvm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--measure-steps', type=int, default=1)
    args, opts = parser.parse_known_args()
    device = args.device
    measure_steps = args.measure_steps
        
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False

    input_fnames = [x for x in os.listdir("mm_files") if x.startswith('in_feat')]
    input_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
    kernel_fnames = [x for x in os.listdir("mm_files") if x.startswith('kernel')]
    kernel_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))

    target = 'cuda'
    ctx = tvm.context(target, 0)
    s, args = matmul_d2l(n=1600, m=64, l=64)
    mod_d2l = tvm.build(s, args, target)
    s, args = matmul_topi(n=1600, m=64, l=64)
    mod_topi = tvm.build(s, args, target)

    time_tvm_d2l_lst = [0] * measure_steps
    time_wallclock_d2l_lst = [0] * measure_steps
    time_tvm_topi_lst = [0] * measure_steps
    time_wallclock_topi_lst = [0] * measure_steps
    time_baseline_lst = [0] * measure_steps

    input_fname, kernel_fname = input_fnames[-1], kernel_fnames[-1]
    in_feat, kernel = load_tensor(input_fname), load_tensor(kernel_fname)

    print(in_feat.shape, kernel.shape)

    with torch.no_grad():
        for idx in range(1 + measure_steps):
            torch.cuda.synchronize()
            st = time.time()
            out_feat = torch.mm(in_feat, kernel)
            torch.cuda.synchronize()
            ed = time.time()
            if idx >= 1:
                time_baseline_lst[idx-1] += (ed-st)

    in_feat, kernel = in_feat.cpu().numpy(), kernel.cpu().numpy()
    out_feat = np.empty_like(np.zeros((in_feat.shape[0], kernel.shape[-1]))).astype(in_feat.dtype)

    for idx in range(1 + measure_steps):
        time_tvm_d2l = bench_matmul_tvm(in_feat, kernel, out_feat, mod_d2l, ctx)
        time_tvm_topi = bench_matmul_tvm(in_feat, kernel.T, out_feat, mod_topi, ctx)

        a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [in_feat, kernel, out_feat]]
        torch.cuda.synchronize()
        st1 = time.time()
        mod_d2l(a, b, c)
        torch.cuda.synchronize()
        ed1 = time.time()

        a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [in_feat, kernel.T, out_feat]]
        torch.cuda.synchronize()
        st2 = time.time()
        mod_topi(a, b, c)
        torch.cuda.synchronize()
        ed2 = time.time()

        if idx >= 1:
            time_wallclock_d2l_lst[idx-1] += (ed1-st1)
            time_tvm_d2l_lst[idx-1] += time_tvm_d2l
            time_wallclock_topi_lst[idx-1] += (ed2-st2)
            time_tvm_topi_lst[idx-1] += time_tvm_topi

    print(f"tvm d2l matmul total time: {sum(time_tvm_d2l_lst)}")
    print(f"tvm d2l matmul total wallclock time: {sum(time_wallclock_d2l_lst)}")
    print(f"tvm topi matmul total time: {sum(time_tvm_topi_lst)}")
    print(f"tvm topi matmul total wallclock time: {sum(time_wallclock_topi_lst)}")
    print(f"torch.mm total time: {sum(time_baseline_lst)}")