# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time DiSCO's optimizer step on a real flavor with synthetic gradients.

    torchrun --nproc_per_node 4 tests/unit_tests/gpu/disco/bench_step.py \
        --flavor mis-8b-proxy --steps 30 --log-norm-freq 10 --dp-shard 4

Reports the median non-logging step and the median logging step (norms armed
every `--log-norm-freq` steps), each as CUDA-event time around
`container.step()`. No data pipeline, no forward/backward: this isolates the
optimizer, which is what the refactor changes.
"""

import argparse
import os
import statistics
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from torchtitan.models.opt_moe import model_registry  # noqa: E402


def _set_grads(model, step: int, device) -> None:
    """Random gradients made on the device, each rank only its own shard: the
    step's kernels do not depend on the values, and building every full
    gradient on the host would take seconds per step on the real 8B."""
    gen = torch.Generator(device=device).manual_seed(step * 7_919 + dist.get_rank())
    for _, p in harness.trainable(model):
        if isinstance(p, DTensor):
            local = p.to_local()
            g = torch.randn(
                local.shape, generator=gen, device=device, dtype=local.dtype
            )
            p.grad = DTensor.from_local(
                g.mul_(1e-2),
                p.device_mesh,
                p.placements,
                shape=p.shape,
                stride=p.stride(),
            )
        else:
            g = torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype)
            p.grad = g.mul_(1e-2)


def _union_ms(intervals) -> float:
    total, end = 0.0, float("-inf")
    for lo, hi in sorted(intervals):
        if hi > end:
            total += hi - max(lo, end)
            end = hi
    return total / 1e3


def _profile_plain_steps(container, model, device, args, rank) -> None:
    """Where a plain step goes. Host: the time to enqueue the step, measured
    without the profiler (a step whose host time reaches its wall time is
    launch-bound). GPU: kernel time by category and the union of all kernels,
    counted only inside the step (the synthetic gradients are set up once,
    before any of it)."""
    from torch.profiler import profile, ProfilerActivity, record_function

    _set_grads(model, args.warmup + args.steps + 1, device)
    grads = {p: p.grad for _, p in harness.trainable(model)}

    def one_step():
        for p, g in grads.items():
            p.grad = g
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        with record_function("bench.step"):
            container.step()
        host = (time.perf_counter() - t0) * 1e3
        end.record()
        torch.cuda.synchronize()
        container.zero_grad()
        return host, start.elapsed_time(end)

    timed = [one_step() for _ in range(args.profile)]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile):
            one_step()
    import cProfile
    import io
    import pstats

    cp = cProfile.Profile()  # every rank steps: the step has collectives
    cp.enable()
    for _ in range(args.profile):
        one_step()
    cp.disable()
    if rank != 0:
        return
    n = args.profile
    # Nested ranges reach the GPU timeline only as the innermost annotation, so
    # a kernel belongs to the step when its launch happened inside the host range.
    events = prof.events()
    spans = [
        (e.time_range.start, e.time_range.end)
        for e in events
        if e.activity_type == "user_annotation" and e.name == "bench.step"
    ]
    launched = {
        e.id: e.time_range.start
        for e in events
        if e.activity_type in ("cuda_runtime", "cuda_driver")
    }
    kernels = [
        e
        for e in events
        if e.activity_type in ("kernel", "gpu_memcpy", "gpu_memset")
        and any(lo <= launched.get(e.id, -1.0) <= hi for lo, hi in spans)
    ]
    cats = {
        "nccl (all-to-all, all-reduce)": ("nccl",),
        "matmul / bmm (orthogonalisers)": (
            "gemm",
            "cutlass",
            "sm90",
            "sm100",
            "bmm",
            "matmul",
            "cublas",
            "xmma",
        ),
        "foreach (momentum, apply)": ("foreach", "multi_tensor"),
        "copy / cast / pack": (
            "copy",
            "cast",
            "elementwise_kernel",
            "unrolled",
            "memcpy",
        ),
        "triton (compiled phases)": ("triton",),
    }
    totals = {k: 0.0 for k in cats}
    other = 0.0
    for e in kernels:
        t = (e.time_range.end - e.time_range.start) / 1e3
        name = e.name.lower()
        for k, keys in cats.items():
            if any(x in name for x in keys):
                totals[k] += t
                break
        else:
            other += t
    host = statistics.median(h for h, _ in timed)
    wall = statistics.median(w for _, w in timed)
    busy = _union_ms((e.time_range.start, e.time_range.end) for e in kernels) / n
    print(
        f"[profile] {n} plain steps without the profiler: wall {wall:.1f} ms, "
        f"host enqueue {host:.1f} ms (median)"
    )
    print(
        f"[profile] with it: {len(spans)} step spans, {len(kernels) / n:.0f} kernels "
        f"per step, GPU busy {busy:.1f} ms per step (union of kernels)"
    )
    for k, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"[profile]   {k:34s} {t / n:7.2f} ms")
    print(f"[profile]   {'other':34s} {other / n:7.2f} ms")
    per_name: dict[str, list[float]] = {}
    for e in kernels:
        per_name.setdefault(e.name, []).append(
            (e.time_range.end - e.time_range.start) / 1e3
        )
    for name, ts in sorted(per_name.items(), key=lambda kv: -sum(kv[1]))[:12]:
        print(f"[profile]     {sum(ts) / n:7.2f} ms  x{len(ts) // n:<4d} {name[:90]}")
    out = io.StringIO()
    pstats.Stats(cp, stream=out).sort_stats("tottime").print_stats(25)
    for line in out.getvalue().splitlines():
        if line.strip():
            print(f"[cprofile] {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", default="mis-8b-proxy")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--log-norm-freq", type=int, default=10)
    ap.add_argument("--dp-shard", type=int, default=4)
    ap.add_argument("--dp-replicate", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--compute", default="gram_polar_express")
    ap.add_argument("--compute-steps", type=int, default=6)
    ap.add_argument("--gram-level", type=int, default=0)
    ap.add_argument("--local-logging", action="store_true")
    ap.add_argument("--comm-dtype", default="float32")
    ap.add_argument("--rule", default="muon", help="the default group's rule")
    ap.add_argument(
        "--profile",
        type=int,
        default=0,
        help="profile this many plain steps after the timed ones and print "
        "where the GPU time goes",
    )
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    pd = harness.parallel_dims(
        world_size=world,
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=args.cp,
        ep=args.ep,
    )
    cfg = harness.trainer_config(
        model_registry(args.flavor),
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=args.cp,
        ep=args.ep,
        compute=args.compute,
        steps=args.compute_steps,
        log_norm_freq=args.log_norm_freq,
        gram_level=args.gram_level,
        communication_dtype=args.comm_dtype,
    )
    cfg.optimizer.groups["default"]["rule"] = args.rule
    with tempfile.TemporaryDirectory() as tmp:
        model, container = harness.build_model_and_optimizer(cfg, pd, device, tmp)
    container.log_metrics_locally = args.local_logging

    n_params = sum(p.numel() for _, p in harness.trainable(model))
    if rank == 0:
        print(
            f"[bench] flavor={args.flavor} params={n_params / 1e6:.1f}M world={world} "
            f"dp_shard={args.dp_shard} dp_replicate={args.dp_replicate} ep={args.ep} cp={args.cp} "
            f"compute={args.compute}/{args.compute_steps} log_norm_freq={args.log_norm_freq} "
            f"local_logging={args.local_logging} comm={args.comm_dtype} rule={args.rule}"
        )

    plain, logging = [], []
    for step in range(1, args.warmup + args.steps + 1):
        _set_grads(model, step, device)
        armed = args.log_norm_freq > 0 and step % args.log_norm_freq == 0
        if armed:
            container.arm_norm_computation()
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        container.step()
        if armed:
            metrics = container.collect_metrics(step)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        container.zero_grad()
        if step <= args.warmup:
            continue
        (logging if armed else plain).append(ms)
        if armed and rank == 0 and len(logging) == 1:
            print(f"[bench] first logging step: {len(metrics)} metric keys")

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    # Every rank's number matters: the step ends when the slowest rank does.
    t = torch.tensor([med(plain), med(logging)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[bench] median step: plain {t[0].item():.1f} ms ({len(plain)} steps), "
            f"logging {t[1].item():.1f} ms ({len(logging)} steps)  [max over ranks]"
        )
    if args.profile:
        try:
            _profile_plain_steps(container, model, device, args, rank)
        except Exception as e:  # the timing above must survive a profiler problem
            print(f"[profile] failed: {type(e).__name__}: {e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
