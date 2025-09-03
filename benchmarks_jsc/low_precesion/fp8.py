# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# these code are AI-Generated
import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


try:  # plotting deps are optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

try:
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig

    TORCHAO_AVAILABLE = True
except Exception:
    convert_to_float8_training = None  # type: ignore
    Float8LinearConfig = None  # type: ignore
    TORCHAO_AVAILABLE = False


def _assert_cuda_bf16():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")
    # BF16 on Ampere+ GPUs is widely available; we don't hard-fail but warn if unsupported
    if not torch.cuda.is_bf16_supported():
        print("Warning: CUDA BF16 not reported as supported on this device.")


def _can_use_fp8_dims(d_in: int, d_out: int) -> bool:
    # FP8 kernels typically require K/N dims divisible by 16 for best perf/compat
    return (d_in % 16 == 0) and (d_out % 16 == 0)


class RMSNormLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # self.rms = torch.nn.RMSNorm(d_in, elementwise_affine=False)
        self.fc = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.rms(x)
        x = self.fc(x)
        return x


def build_model(
    d_in: int,
    d_out: int,
    precision: str,
    recipe: Optional[str],
    use_compile: bool,
    device: torch.device,
) -> nn.Module:
    # Use RMSNorm (no parameters) followed by Linear, per request
    m = RMSNormLinear(d_in, d_out).to(device).bfloat16()

    if precision == "fp8":
        if not TORCHAO_AVAILABLE:
            raise RuntimeError(
                "torchao.float8 is not available. Install torchao to run FP8 benchmarks."
            )
        if recipe is None:
            raise ValueError("FP8 precision requires a recipe name.")
        if not _can_use_fp8_dims(d_in, d_out):
            raise ValueError(
                f"FP8 requires d_in and d_out divisible by 16, got {d_in}, {d_out}"
            )

        # Configure and convert
        cfg = Float8LinearConfig.from_recipe_name(recipe)  # type: ignore[attr-defined]
        m = convert_to_float8_training(m, config=cfg)

    # Optional compile for performance
    if use_compile:
        try:
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            m = torch.compile(m)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Warning: torch.compile unavailable or failed: {e}")
    return m


@torch.inference_mode(False)
def benchmark_once(
    d_in: int,
    d_out: int,
    bs: int,
    seq_len: int,
    precision: str,
    recipe: Optional[str],
    steps: int,
    warmup: int,
    use_compile: bool,
    device: torch.device,
    lr: float = 0.1,
) -> Tuple[float, float]:
    """
    Measures GPU kernel time for fwd+bwd of RMSNorm->Linear.
    Returns tokens/sec and avg step time (ms) based on GPU event timing.
    """
    model = build_model(d_in, d_out, precision, recipe, use_compile, device)

    M = bs * seq_len
    x = torch.randn(M, d_in, device=device, dtype=torch.bfloat16).requires_grad_()
    grad_out = torch.randn(M, d_out, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(max(warmup, 0)):
        y = model(x)
        y.backward(grad_out)
        if x.grad is not None:
            x.grad = None
    torch.cuda.synchronize()

    # GPU event timing for measured steps
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(steps):
        y = model(x)
        y.backward(grad_out)
        if x.grad is not None:
            x.grad = None
    end_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(end_ev)

    elapsed_s = elapsed_ms / 1000.0
    total_tokens = bs * seq_len * steps
    throughput = total_tokens / elapsed_s if elapsed_s > 0 else float("nan")
    avg_step_ms = elapsed_ms / steps if steps > 0 else float("nan")
    return throughput, avg_step_ms


def run_grid(
    d_in_list: List[int],
    d_out_list: List[int],
    bs_list: List[int],
    seq_len: int,
    steps: int,
    warmup: int,
    use_compile: bool,
    device: torch.device,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    precisions = ["bf16", "fp8"]
    recipes = [None, "tensorwise", "rowwise", "rowwise_with_gw_hp"]

    for d_in in d_in_list:
        for d_out in d_out_list:
            for bs in bs_list:
                for prec in precisions:
                    if prec == "bf16":
                        recipe = None
                        try:
                            thr, step_ms = benchmark_once(
                                d_in,
                                d_out,
                                bs,
                                seq_len,
                                precision=prec,
                                recipe=recipe,
                                steps=steps,
                                warmup=warmup,
                                use_compile=use_compile,
                                device=device,
                            )
                        except Exception as e:
                            print(
                                f"BF16 benchmark failed for d_in={d_in}, d_out={d_out}, bs={bs}: {e}"
                            )
                            thr, step_ms = float("nan"), float("nan")
                        results.append(
                            {
                                "precision": prec,
                                "recipe": recipe or "-",
                                "d_in": d_in,
                                "d_out": d_out,
                                "bs": bs,
                                "seq_len": seq_len,
                                "throughput_tokens_per_s": thr,
                                "avg_step_ms": step_ms,
                            }
                        )
                    else:
                        for recipe in ["tensorwise", "rowwise", "rowwise_with_gw_hp"]:
                            if not _can_use_fp8_dims(d_in, d_out):
                                # record but skip running
                                results.append(
                                    {
                                        "precision": prec,
                                        "recipe": recipe,
                                        "d_in": d_in,
                                        "d_out": d_out,
                                        "bs": bs,
                                        "seq_len": seq_len,
                                        "throughput_tokens_per_s": float("nan"),
                                        "avg_step_ms": float("nan"),
                                    }
                                )
                                continue
                            try:
                                thr, step_ms = benchmark_once(
                                    d_in,
                                    d_out,
                                    bs,
                                    seq_len,
                                    precision=prec,
                                    recipe=recipe,
                                    steps=steps,
                                    warmup=warmup,
                                    use_compile=use_compile,
                                    device=device,
                                )
                            except Exception as e:
                                print(
                                    f"FP8[{recipe}] benchmark failed for d_in={d_in}, d_out={d_out}, bs={bs}: {e}"
                                )
                                thr, step_ms = float("nan"), float("nan")
                            results.append(
                                {
                                    "precision": prec,
                                    "recipe": recipe,
                                    "d_in": d_in,
                                    "d_out": d_out,
                                    "bs": bs,
                                    "seq_len": seq_len,
                                    "throughput_tokens_per_s": thr,
                                    "avg_step_ms": step_ms,
                                }
                            )
    return results


def save_results(out_dir: str, results: List[Dict[str, object]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sysname = os.environ.get("SYSTEMNAME", "unknown")
    tag = f"{sysname}__" if sysname else ""

    # Detailed results CSV
    detailed_path = os.path.join(out_dir, f"{tag}throughput_detailed.csv")
    fieldnames = [
        "precision",
        "recipe",
        "d_in",
        "d_out",
        "bs",
        "seq_len",
        "throughput_tokens_per_s",
        "avg_step_ms",
    ]
    with open(detailed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    # Compute BF16 baselines for speedup
    bf16_index: Dict[Tuple[int, int, int, int], float] = {}
    for r in results:
        if r["precision"] == "bf16":
            key = (int(r["d_in"]), int(r["d_out"]), int(r["bs"]), int(r["seq_len"]))
            bf16_index[key] = (
                float(r["throughput_tokens_per_s"])
                if r["throughput_tokens_per_s"] == r["throughput_tokens_per_s"]
                else float("nan")
            )

    # Speedup CSV
    speedup_rows: List[Dict[str, object]] = []
    for r in results:
        if r["precision"] == "bf16":
            continue
        key = (int(r["d_in"]), int(r["d_out"]), int(r["bs"]), int(r["seq_len"]))
        base = bf16_index.get(key, float("nan"))
        thr = (
            float(r["throughput_tokens_per_s"])
            if r["throughput_tokens_per_s"] == r["throughput_tokens_per_s"]
            else float("nan")
        )
        speedup = (thr / base) if (base and base == base) else float("nan")
        speedup_rows.append(
            {
                "recipe": r["recipe"],
                "d_in": r["d_in"],
                "d_out": r["d_out"],
                "bs": r["bs"],
                "seq_len": r["seq_len"],
                "speedup_vs_bf16": speedup,
            }
        )

    speedup_path = os.path.join(out_dir, f"{tag}speedup_vs_bf16.csv")
    with open(speedup_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["recipe", "d_in", "d_out", "bs", "seq_len", "speedup_vs_bf16"],
        )
        w.writeheader()
        for row in speedup_rows:
            w.writerow(row)

    # Per-recipe matrices: rows = bs, cols = f"{d_in}x{d_out}"
    by_recipe: Dict[str, Dict[Tuple[int, int, int], float]] = {}
    for r in speedup_rows:
        recipe = str(r["recipe"])  # e.g., tensorwise
        d_in = int(r["d_in"])  # type: ignore[arg-type]
        d_out = int(r["d_out"])  # type: ignore[arg-type]
        bs = int(r["bs"])  # type: ignore[arg-type]
        key = (d_in, d_out, bs)
        by_recipe.setdefault(recipe, {})[key] = float(r["speedup_vs_bf16"])  # type: ignore[arg-type]

    dims_labels = sorted({(int(r["d_in"]), int(r["d_out"])) for r in speedup_rows})
    bs_vals = sorted({int(r["bs"]) for r in speedup_rows})

    for recipe, data in by_recipe.items():
        matrix_path = os.path.join(out_dir, f"{tag}speedup_matrix_{recipe}.csv")
        with open(matrix_path, "w", newline="") as f:
            # header: bs, then each dim label
            header = ["bs"] + [f"{d_in}x{d_out}" for (d_in, d_out) in dims_labels]
            w = csv.writer(f)
            w.writerow(header)
            for bs in bs_vals:
                row: List[object] = [bs]
                for d_in, d_out in dims_labels:
                    val = data.get((d_in, d_out, bs), float("nan"))
                    row.append(val)
                w.writerow(row)

    print(f"Saved detailed throughput to: {detailed_path}")
    print(f"Saved speedups to: {speedup_path}")
    print(f"Saved per-recipe matrices to: {out_dir}/{tag}speedup_matrix_*.csv")

    # Plot heatmaps per recipe if plotting is available
    if HAS_PLOTTING:
        try:
            plot_speedup_heatmaps(out_dir, results)
        except Exception as e:
            print(f"Warning: plotting failed: {e}")
    else:
        print("matplotlib not available; skipping heatmap plots.")


def _compute_speedups_from_results(results: List[Dict[str, object]]):
    # Build bf16 baseline
    bf16_index: Dict[Tuple[int, int, int, int], float] = {}
    for r in results:
        if r["precision"] == "bf16":
            key = (int(r["d_in"]), int(r["d_out"]), int(r["bs"]), int(r["seq_len"]))
            val = (
                float(r["throughput_tokens_per_s"])
                if r["throughput_tokens_per_s"] == r["throughput_tokens_per_s"]
                else float("nan")
            )
            bf16_index[key] = val

    # Compute per-recipe speedups
    by_recipe: Dict[str, Dict[Tuple[int, int, int], float]] = {}
    d_in_set, d_out_set, bs_set, seq_set = set(), set(), set(), set()
    for r in results:
        if r["precision"] != "bf16":
            recipe = str(r["recipe"])  # type: ignore
            d_in = int(r["d_in"])  # type: ignore
            d_out = int(r["d_out"])  # type: ignore
            bs = int(r["bs"])  # type: ignore
            seq = int(r["seq_len"])  # type: ignore
            base = bf16_index.get((d_in, d_out, bs, seq), float("nan"))
            thr = (
                float(r["throughput_tokens_per_s"])
                if r["throughput_tokens_per_s"] == r["throughput_tokens_per_s"]
                else float("nan")
            )
            val = (thr / base) if (base and base == base) else float("nan")
            by_recipe.setdefault(recipe, {})[(d_in, d_out, bs)] = val
            d_in_set.add(d_in)
            d_out_set.add(d_out)
            bs_set.add(bs)
            seq_set.add(seq)
    return (
        by_recipe,
        sorted(d_in_set),
        sorted(d_out_set),
        sorted(bs_set),
        sorted(seq_set),
    )


def plot_speedup_heatmaps(out_dir: str, results: List[Dict[str, object]]):
    (
        by_recipe,
        d_in_list,
        d_out_list,
        bs_list,
        seq_list,
    ) = _compute_speedups_from_results(results)
    if len(seq_list) != 1:
        print(
            f"Multiple seq_len values detected: {seq_list}. Using first for labeling."
        )
    seq_len = seq_list[0] if seq_list else 4096

    sysname = os.environ.get("SYSTEMNAME", "unknown")
    tag = f"{sysname}__" if sysname else ""

    for recipe, data in by_recipe.items():
        # Build tall matrix stacking blocks per tokens=bs*seq
        rows: List[List[float]] = []
        row_labels: List[str] = []
        block_height = len(d_in_list)
        for bs in bs_list:
            tokens = bs * seq_len
            for d_in in d_in_list:
                row = [
                    data.get((d_in, d_out, bs), float("nan")) for d_out in d_out_list
                ]
                rows.append(row)
                # Label the first row of each block with the token count + d_in
                if d_in == d_in_list[0]:
                    row_labels.append(f"T={tokens}, {d_in}")
                else:
                    row_labels.append(str(d_in))

        mat = np.array(rows, dtype=float) if rows else np.zeros((0, 0), dtype=float)

        # Create RGBA image using sign-based colors and alpha for magnitude
        H, W = mat.shape
        if H == 0 or W == 0:
            continue

        rgba = np.ones((H, W, 4), dtype=float)
        # base colors
        green = np.array([0.0, 0.6, 0.0])
        red = np.array([0.8, 0.0, 0.0])
        gray = np.array([0.7, 0.7, 0.7])

        # Compute alphas from |log2(speedup)|, clamp to percentile to avoid outliers
        vals = mat[np.isfinite(mat) & (mat > 0)]
        if vals.size:
            mags = np.abs(np.log2(vals))
            # clamp alpha scale to 95th percentile or at least 1.0 (i.e., 2x)
            max_mag = max(np.percentile(mags, 95), 1.0)
        else:
            max_mag = 1.0

        for i in range(H):
            for j in range(W):
                v = mat[i, j]
                if not np.isfinite(v) or v <= 0:
                    rgba[i, j, :3] = gray
                    rgba[i, j, 3] = 0.3
                    continue
                if math.isclose(v, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                    # near parity -> nearly transparent
                    base = green  # arbitrary; alpha ~0 will make it invisible
                    a = 0.05
                elif v > 1.0:
                    base = green
                    a = min(abs(math.log2(v)) / max_mag, 1.0)
                else:
                    base = red
                    a = min(abs(math.log2(v)) / max_mag, 1.0)
                rgba[i, j, :3] = base
                rgba[i, j, 3] = a

        # Plot
        fig_w = max(6.0, 0.6 * W + 2.0)
        fig_h = max(4.0, 0.35 * H + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        ax.imshow(rgba, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(W))
        ax.set_xticklabels([str(d) for d in d_out_list], rotation=45, ha="right")
        ax.set_xlabel("d_out")

        ax.set_yticks(range(H))
        ax.set_yticklabels(row_labels)
        ax.set_ylabel("d_in")

        # Draw block separators between different bs groups
        for k in range(1, len(bs_list)):
            y = k * block_height - 0.5
            ax.axhline(y, color="black", linewidth=0.5)
        # No separate token labels; encoded in the first row label of each block

        # Per-cell text with speedup value
        for i in range(H):
            for j in range(W):
                v = mat[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    alpha=0.9,
                )

        # Title with system info
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        try:
            import torchao as _torchao_pkg

            torchao_ver = getattr(_torchao_pkg, "__version__", "unknown")
        except Exception:
            torchao_ver = "unavailable"
        subtitle = f"{sysname} | GPU {gpu_name} | PyTorch {torch.__version__}+cu{torch.version.cuda} | torchao {torchao_ver}"
        ax.set_title(
            f"FP8 speedup vs BF16 — RMSNorm->Linear fwd+bwd — recipe={recipe}\n{subtitle}"
        )
        ax.grid(False)

        # Add a small legend-like annotation
        ax.text(
            1.02,
            1.0,
            "Green: speedup\nRed: slowdown\nAlpha ∝ |log2(speedup)|",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

        # Tighten layout with a standard left margin now that labels are compact
        plt.tight_layout(rect=[0.12, 0.06, 0.98, 0.95])
        out_path = os.path.join(out_dir, f"{tag}speedup_heatmap_{recipe}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved heatmap: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BF16 vs FP8 Linear training throughput benchmark"
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length (fixed: suggested 4096)",
    )
    p.add_argument(
        "--max-bs",
        type=int,
        default=10,
        help="Max batch size so that bs*seq_len <= 40960",
    )
    p.add_argument(
        "--d-in-list",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated d_in list (multiples of 16)",
    )
    p.add_argument(
        "--d-out-list",
        type=str,
        default="2048,4096,8192, 16384",
        help="Comma-separated d_out list (multiples of 16)",
    )
    p.add_argument("--steps", type=int, default=20, help="Measured steps per config")
    p.add_argument("--warmup", type=int, default=5, help="Warmup steps per config")
    p.add_argument(
        "--out-dir",
        type=str,
        default="./fp8_bench_results",
        help="Output directory for CSVs",
    )
    return p.parse_args()


def main() -> None:
    _assert_cuda_bf16()
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = parse_args()

    seq_len = args.seq_len
    max_tokens = 40960  # cap as per request (bs <= 10 when seq_len=4096)
    max_bs = min(args.max_bs, max_tokens // seq_len)
    bs_list = list(range(1, max_bs + 1))

    d_in_list = [int(x) for x in args.d_in_list.split(",") if x.strip()]
    d_out_list = [int(x) for x in args.d_out_list.split(",") if x.strip()]

    # Ensure divisibility by 16 for FP8-friendly configs (we still run BF16 otherwise)
    if not all((x % 16 == 0) for x in d_in_list + d_out_list):
        print(
            "Warning: Some dims are not divisible by 16; FP8 results will be NaN for those."
        )

    device = torch.device("cuda")
    results = run_grid(
        d_in_list=d_in_list,
        d_out_list=d_out_list,
        bs_list=bs_list,
        seq_len=seq_len,
        steps=args.steps,
        warmup=args.warmup,
        use_compile=True,
        device=device,
    )

    save_results(args.out_dir, results)


if __name__ == "__main__":
    main()
    # python torchtitan/benchmarks_jsc/low_precesion/fp8.py --out-dir ./benchmarks/fp8
