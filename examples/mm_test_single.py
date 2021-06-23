import os
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
import tvm
from tvm import te, autotvm

from utils import load_tensor
from mm_test import bench_matmul_tvm
from autotvm_test import matmul_topi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--measure-steps', type=int, default=1)
    parser.add_argument('--num-repeats', type=int, default=1000)
    parser.add_argument('--tuned', action='store_true')
    parser.add_argument('--id', type=int, default=0)
    args, opts = parser.parse_known_args()
    device = args.device
    measure_steps = args.measure_steps
    num_repeats = args.num_repeats
        
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False

    input_fnames = [x for x in os.listdir("mm_files") if x.startswith('in_feat')]
    input_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
    kernel_fnames = [x for x in os.listdir("mm_files") if x.startswith('kernel')]
    kernel_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))

    # input_fname, kernel_fname = input_fnames[-1], kernel_fnames[-1]
    input_fname, kernel_fname = input_fnames[args.id], kernel_fnames[args.id]
    in_feat, kernel = load_tensor(input_fname), load_tensor(kernel_fname)
    print(in_feat.shape, kernel.shape)

    n, m, l = in_feat.shape[0], kernel.shape[1], in_feat.shape[1]

    target = 'cuda'
    ctx = tvm.context(target, 0)

    if args.tuned:
        from autotvm_test import matmul_d2l
        with autotvm.apply_history_best("matmul_d2l_1600.log"):
            with tvm.target.Target("cuda"):
                s, args = matmul_d2l(n, m, l)
                mod_d2l = tvm.build(s, args)
        
        with autotvm.apply_history_best("matmul_topi.log"):
            with tvm.target.Target("cuda"):
                s, arg_bufs = matmul_topi(n, m, l)
                mod_topi = tvm.build(s, arg_bufs)
    else:
        from mm_test import matmul_d2l
        s, args = matmul_d2l(n=n, m=m, l=l)
        mod_d2l = tvm.build(s, args, target)

        s, arg_bufs = matmul_topi(n, m, l)
        mod_topi = tvm.build(s, arg_bufs, target)


    time_cudasync_lst = [0] * measure_steps
    time_tvm_d2l_lst = [0] * measure_steps
    time_wallclock_d2l_lst = [0] * measure_steps
    time_tvm_topi_lst = [0] * measure_steps
    time_wallclock_topi_lst = [0] * measure_steps
    time_baseline_lst = [0] * measure_steps

    with torch.no_grad():
        for idx in range(1 + measure_steps):
            torch.cuda.synchronize()
            st = time.time()
            for i in range(num_repeats):
                out_feat = torch.mm(in_feat, kernel)
            torch.cuda.synchronize()
            ed = time.time()
            if idx >= 1:
                time_baseline_lst[idx-1] += (ed-st)

    in_feat, kernel = in_feat.cpu().numpy(), kernel.cpu().numpy()
    out_feat = np.empty_like(np.zeros((in_feat.shape[0], kernel.shape[-1]))).astype(in_feat.dtype)

    for idx in range(1 + measure_steps):
        # record tvm kernel time
        time_tvm_d2l = bench_matmul_tvm(in_feat, kernel, out_feat, mod_d2l, ctx, num_repeats)
        time_tvm_topi = bench_matmul_tvm(in_feat, kernel.T, out_feat, mod_topi, ctx, num_repeats)

        # record cuda sync overhead
        torch.cuda.synchronize()
        st0 = time.time()
        torch.cuda.synchronize()
        ed0 = time.time()

        # record tvm wallclock time
        a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [in_feat, kernel, out_feat]]
        torch.cuda.synchronize()
        st1 = time.time()
        for i in range(num_repeats):
            mod_d2l(a, b, c)
        torch.cuda.synchronize()
        ed1 = time.time()

        a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [in_feat, kernel.T, out_feat]]
        torch.cuda.synchronize()
        st2 = time.time()
        for i in range(num_repeats):
            mod_topi(a, b, c)
        torch.cuda.synchronize()
        ed2 = time.time()

        if idx >= 1:
            time_cudasync_lst[idx-1] += (ed0-st0)
            time_wallclock_d2l_lst[idx-1] += (ed1-st1)
            time_tvm_d2l_lst[idx-1] += time_tvm_d2l
            time_wallclock_topi_lst[idx-1] += (ed2-st2)
            time_tvm_topi_lst[idx-1] += time_tvm_topi

    print(f"cuda sync total time: {np.mean(time_cudasync_lst)} ± {np.std(time_cudasync_lst)}")
    print(f"tvm d2l matmul total time: {np.mean(time_tvm_d2l_lst)} ± {np.std(time_tvm_d2l_lst)}")
    print(f"tvm d2l matmul total wallclock time: {np.mean(time_wallclock_d2l_lst)} ± {np.std(time_wallclock_d2l_lst)}")
    print(f"tvm topi matmul total time: {np.mean(time_tvm_topi_lst)} ± {np.std(time_tvm_topi_lst)}")
    print(f"tvm topi matmul total wallclock time: {np.mean(time_wallclock_topi_lst)} ± {np.std(time_wallclock_topi_lst)}")
    print(f"torch.mm total time: {np.mean(time_baseline_lst)} ± {np.std(time_baseline_lst)}")

    # save data
    np.save('time_wallclock_d2l_lst.npy', time_wallclock_d2l_lst)
    np.save('time_tvm_d2l_lst.npy', time_tvm_d2l_lst)
    np.save('time_baseline_lst.npy', time_baseline_lst)