import os
from tqdm import tqdm
import numpy as np
import argparse
import torch
import time
import tvm
from tvm import te


def load_tensor(fname, dir="mm_files"):
    model = torch.jit.load(os.path.join(dir, fname))
    x = list(model.parameters())[0]
    return x.detach()


def read_mm_time(fname="mm_time.txt"):
    with open(fname, "r") as f:
        lines = f.readlines()
    times = [float(x.split()[-1][:-1]) for x in lines]
    return sum(times)


def matmul():
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    n, m, l = te.var(name='n'), te.var(name='m'), te.var(name='l')
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return A, B, C


def split(stage, axis, factors):
    """Split an axis by a list of factors in a reverse order
    """
    axes = []
    for f in reversed(factors):
        axis, x = stage.split(axis, f)
        axes.append(x)
    return list(reversed(axes+[axis]))


def bind_thread(stage, axes, tags):
    """Bind a list of axes to thread axes
    """
    for axis, tag in zip(axes, tags):
        stage.bind(axis, te.thread_axis(tag))


def bench_workload(workload):
    """Benchmark a workload

    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats


def bench_matmul_tvm(i, k, o, mod, ctx):
    def workload(nrepeats):
        timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
        return timer(a, b, c).mean * nrepeats
    a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [i, k, o]]
    # time = bench_workload(workload)
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    return time

 
def matmul_gpu(block_size=16, tx=8, ty=4, tk=32):
    A, B, C = matmul()
    s = te.create_schedule(C.op)
    # Create caches
    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[C].op.axis
    xb, xo, xi = split(s[C], x, (block_size, tx))
    yb, yo, yi = split(s[C], y, (block_size, ty))
    s[C].reorder(xb, yb, xo, yo, xi, yi)
    # Note that we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[C], (yb, xb, yo, xo),
                ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # Schedule C_local
    s[C_local].compute_at(s[C], yo)
    yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis
    ko, ki = s[C_local].split(k, tk)
    s[C_local].reorder(ko, ki, yi, xi)
    # Optimize read caches of A and B with cooperative fetching
    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        y, x = s[shared].op.axis
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, yi = s[shared].split(y, nparts=block_size)
        xo, xi = s[shared].split(x, nparts=block_size)
        s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))
    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    return s, (A, B, C)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--measure-steps', type=int, default=1)
    args, opts = parser.parse_known_args()
    device = args.device
    measure_steps = args.measure_steps
        
    torch.manual_seed(1)

    input_fnames = [x for x in os.listdir("mm_files") if x.startswith('in_feat')]
    input_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
    kernel_fnames = [x for x in os.listdir("mm_files") if x.startswith('kernel')]
    kernel_fnames.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))

    s, args = matmul_gpu()
    target = 'cuda'
    mod = tvm.build(s, args, target)
    ctx = tvm.context(target, 0)

    time_tvm_lst = [0] * measure_steps
    time_wallclock_lst = [0] * measure_steps
    time_baseline_lst = [0] * measure_steps
    for i in tqdm(range(len(input_fnames))):
        input_fname, kernel_fname = input_fnames[i], kernel_fnames[i]
        in_feat, kernel = load_tensor(input_fname), load_tensor(kernel_fname)

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
        timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=1)

        for idx in range(1 + measure_steps):
            a, b, c = [tvm.nd.array(x, ctx=ctx) for x in [in_feat, kernel, out_feat]]
            torch.cuda.synchronize()
            st = time.time()
            mod(a, b, c)
            torch.cuda.synchronize()
            ed = time.time()
            time_tvm = bench_matmul_tvm(in_feat, kernel, out_feat, mod, ctx)
            if idx >= 1:
                time_wallclock_lst[idx-1] += (ed-st)
                time_tvm_lst[idx-1] += time_tvm

    print(f"tvm matmul total time: {np.mean(time_tvm_lst)} ± {np.std(time_tvm_lst)}")
    print(f"tvm matmul total wallclock time: {np.mean(time_wallclock_lst)} ± {np.std(time_wallclock_lst)}")
    print(f"torch.mm total time: {np.mean(time_baseline_lst)} ± {np.std(time_baseline_lst)}")
    print(f"torch:mm_out total time {read_mm_time()}")