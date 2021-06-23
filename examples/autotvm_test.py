import logging
import sys
import numpy as np

import tvm
from tvm import te, topi, testing
import tvm.testing
from tvm import autotvm

from mm_test import bind_thread, split


@autotvm.template("matmul_topi")
def matmul_topi(n=None, m=None, l=None):
    if n == None:
        n = te.var(name='n')
    if m == None:
        m = te.var(name='m')
    if l == None:
        l = te.var(name='l')

    A = te.placeholder((n, l), name='A')
    B = te.placeholder((m, l), name='B')
    with tvm.target.cuda():
        if n < 32:
            C = topi.cuda.dense_small_batch(A, B)
            s = topi.cuda.schedule_dense_small_batch(C)
        else:
            C = topi.cuda.dense_large_batch(A, B)
            s = topi.cuda.schedule_dense_large_batch(C)
    return s, [A, B, C]


@autotvm.template("matmul_d2l")
def matmul_d2l(n=None, m=None, l=None):
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')

    
    s = te.create_schedule(C.op)
    cfg = autotvm.get_config()

    cfg.define_knob("block_size", [4, 8, 16, 32, 64])

    # Create caches
    A_shared = s.cache_read(A, "shared", [C])
    A_local  = s.cache_read(A_shared, "local", [C])
    B_shared = s.cache_read(B, "shared", [C])
    B_local  = s.cache_read(B_shared, "local", [C])
    C_local = s.cache_write(C, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[C].op.axis
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=2)
    xbo, xi = cfg["tile_x"].apply(s, C, x)
    ybo, yi = cfg["tile_y"].apply(s, C, y)
    xb, xo = s[C].split(xbo, cfg["block_size"].val)
    yb, yo = s[C].split(ybo, cfg["block_size"].val)
    # xb, xo, xi = split(s[C], x, (block_size, tx))
    # yb, yo, yi = split(s[C], y, (block_size, ty))
    s[C].reorder(xb, yb, xo, yo, xi, yi)

    # Note that we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[C], (yb, xb, yo, xo),
                ("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # Schedule C_local
    s[C_local].compute_at(s[C], yo)
    yi, xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis

    cfg.define_split("tile_k", k, num_outputs=2)
    ko, ki = cfg["tile_k"].apply(s, C_local, k)
    # ko, ki = s[C_local].split(k, tk)
    s[C_local].reorder(ko, ki, yi, xi)
    # Optimize read caches of A and B with cooperative fetching
    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        y, x = s[shared].op.axis
        # Note that we must split into block_size parts to reuse
        # the previous axis threads
        yo, yi = s[shared].split(y, nparts=cfg["block_size"].val)
        xo, xi = s[shared].split(x, nparts=cfg["block_size"].val)
        # yo, yi = s[shared].split(y, nparts=block_size)
        # xo, xi = s[shared].split(x, nparts=block_size)
        s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo, xo), ("threadIdx.y", "threadIdx.x"))
    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    return s, (A, B, C)


if __name__ == '__main__':
    # logging config (for printing tuning log to screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    n, m, l = 1600, 64, 64
    # n, m, l = 3700, 128, 192
    task = autotvm.task.create(
        "matmul_d2l", args=(n, m, l), target="cuda"
    )
    print(task.config_space)

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul_d2l.log")],
    )


    # inspect the best config
    dispatch_context = autotvm.apply_history_best("matmul_d2l.log")
    best_config = dispatch_context.query(task.target, task.workload)
    print(task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best("matmul_d2l.log"):
        with tvm.target.Target("cuda"):
            s, arg_bufs = matmul_d2l(n, m, l)
            func = tvm.build(s, arg_bufs)

    # # check correctness
    f_np = np.random.uniform(size=(n, l)).astype(np.float32)
    k_np = np.random.uniform(size=(l, m)).astype(np.float32)
    c_np = f_np @ k_np
 
    ctx = tvm.gpu()
    f_tvm = tvm.nd.array(f_np, ctx=ctx)
    # k_tvm = tvm.nd.array(k_np.T, ctx=ctx)
    k_tvm = tvm.nd.array(k_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(f_tvm, k_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)


    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
    print(f"Time cost of this operator: {evaluator(f_tvm, k_tvm, c_tvm).mean * 1000}")