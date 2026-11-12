# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""expert_gemm="sonicmoe": SonicMoE's kernels on the router's selection.

- The routed experts at the production dims (hidden 2048, 256 experts, top 8,
  expert hidden 512, bf16; 4,096 tokens) with the same weights, input, scores
  and experts through the DeepGEMM path and the SonicMoE path (both memory
  modes, inside a rematerialisation checkpoint as a block runs them): output,
  input gradient, both expert-weight gradients and the score gradient agree, and
  both sit at the bf16 level around an fp32 reference.
- The whole OptMoE layer (router, experts, shared expert): SonicMoE vs DeepGEMM.
- Inside a compiled block (``compile_blocks``): SonicMoE's kernels are opaque to
  the compiler (never traced), the compiled block matches eager, and the
  rematerialisation replays the up projection only.
- The flavor's build-time checks and the per-layer memory mode.

The kernel tests need a GPU, ``deep_gemm`` and SONICMOE_SITE (the folder that
holds the sonicmoe package; it is not installed in the environment).

Legend: T tokens, D model dim, E experts, K top-k, F expert hidden dim.
"""

import gc
import os

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.models.base.module import Module
from torchtitan.models.opt_moe import build_model_config

T, D, E, K, F = 4096, 2048, 256, 8, 512
# the recipe's recompute patterns for the other backends' experts; the sonicmoe
# region decides itself (its config flag)
REMAT = ActivationRematConfig(
    save_regions=["*"],
    recompute_regions=["*routed_experts.w13.*", "*routed_experts.w2.*"],
)

needs_sonic = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SONICMOE_SITE"),
    reason="needs a GPU and SONICMOE_SITE",
)


def _flavor(**paths):
    """The production flavor's dims (mis-30b-a3b): one dense and one MoE layer."""
    return build_model_config(
        "mis-30b-a3b",
        parallelism=ParallelismConfig(),
        seq_len=T,
        n_layers=2,
        swa_pattern="FF",
        rope_pattern="NN",
        dense_hidden_dim=6144,
        vocab_size=1024,
        router_topk="triton",
        **paths,
    )


def _build(config, device_dtype=torch.bfloat16):
    with torch.device("meta"):
        module = config.build()
    module.to(device_dtype)
    module.to_empty(device="cuda")
    module.init_states(buffer_device=torch.device("cuda"), init_parameters=False)
    return module.train()


def _fill(param, generator, scale=0.02):
    """In place, deterministic, one slice at a time (no fp32 copy of a whole weight)."""
    with torch.no_grad():
        for i in range(param.shape[0]) if param.dim() > 2 else [slice(None)]:
            chunk = torch.randn(param[i].shape, generator=generator, device="cuda")
            param[i] = (chunk * scale).bfloat16().to(param.dtype)


class _Block(Module):
    def __init__(self, module, name):
        super().__init__()
        setattr(self, name, module)
        self.name = name

    def forward(self, *args):
        return getattr(self, self.name)(*args)


class _Model(Module):
    def __init__(self, module, name):
        super().__init__()
        self.layers = torch.nn.ModuleDict({"1": _Block(module, name)})


def _block(module, name="routed_experts", *, compile=False):
    """`module` as the model runs a transformer block: the recipe's remat
    checkpoint around it (``ActivationRematConfig.apply``) and, with `compile`,
    ``compile_blocks``' fullgraph torch.compile on top."""
    model = _Model(module, name)
    REMAT.apply(model)
    block = model.layers["1"]
    if compile:
        block.compile(fullgraph=True)
    return block


def _experts(expert_gemm, sonic_state="recompute"):
    paths = dict(token_dispatch="gather", expert_gemm=expert_gemm)
    if expert_gemm == "sonicmoe":
        paths["sonic_state"] = sonic_state
    experts = _build(_flavor(**paths).layers[1].feed_forward.routed_experts)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    _fill(experts.w13.weight, generator)
    _fill(experts.w2.weight, generator)
    return experts


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(T, D, generator=g, device="cuda").bfloat16()
    ids = torch.rand(T, E, generator=g, device="cuda").topk(K, dim=-1).indices
    scores = torch.softmax(torch.randn(T, K, generator=g, device="cuda"), dim=-1)
    cotangent = torch.randn(T, D, generator=g, device="cuda").bfloat16()
    return x, ids, scores, cotangent


def _cpu(t):
    return t.detach().float().cpu()


def _run(experts):
    """Forward and backward of one block with the routing of `_inputs`; the
    tensors on the CPU and the bytes the forward holds for the backward."""
    x, ids, scores, cotangent = _inputs()
    x.requires_grad_(True)
    scores.requires_grad_(True)
    rmap = torch.zeros(T, E, dtype=torch.bool, device="cuda").scatter_(1, ids, True)
    block = _block(experts)
    gc.collect()  # the previous run's module is in a reference cycle (its block)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    y = block(x, scores, ids, rmap)
    held = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
    y.backward(cotangent)
    result = dict(
        out=_cpu(y),
        dx=_cpu(x.grad),
        dw13=_cpu(experts.w13.weight.grad.view(E, 2 * F, D)),
        dw2=_cpu(experts.w2.weight.grad),
        dscores=_cpu(scores.grad),
        held_MiB=held / 2**20,
    )
    del y, block
    return result


def _reference():
    """The same math in plain fp32 PyTorch (an expert at a time), autograd."""
    x0, ids, scores0, cotangent = _inputs()
    generator = torch.Generator(device="cuda").manual_seed(1234)
    w13 = torch.empty(E, 2 * F, D, device="cuda")
    w2 = torch.empty(E, D, F, device="cuda")
    _fill(w13, generator)
    _fill(w2, generator)
    x = x0.float().requires_grad_(True)
    scores = scores0.clone().requires_grad_(True)
    w13.requires_grad_(True)
    w2.requires_grad_(True)
    out = torch.zeros(T, D, device="cuda")
    for e in range(E):
        token, slot = (ids == e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        gate_up = x[token] @ w13[e].T
        hidden = torch.nn.functional.silu(gate_up[:, :F]) * gate_up[:, F:]
        out = out.index_add(0, token, (hidden @ w2[e].T) * scores[token, slot][:, None])
    out.backward(cotangent.float())
    return dict(
        out=_cpu(out),
        dx=_cpu(x.grad),
        dw13=_cpu(w13.grad),
        dw2=_cpu(w2.grad),
        dscores=_cpu(scores.grad),
    )


def _rel(a, b):
    return (
        torch.linalg.vector_norm(a - b, dtype=torch.float64)
        / torch.linalg.vector_norm(b, dtype=torch.float64)
    ).item()


TENSORS = ("out", "dx", "dw13", "dw2", "dscores")


@pytest.fixture(scope="module")
def runs():
    pytest.importorskip("deep_gemm")
    reference = _reference()
    torch.cuda.empty_cache()
    results = {}
    for name, args in (
        ("deepgemm", ("deepgemm",)),
        ("sonic recompute", ("sonicmoe", "recompute")),
        ("sonic store", ("sonicmoe", "store")),
    ):
        torch.cuda.reset_peak_memory_stats()
        experts = _experts(*args)
        results[name] = _run(experts)
        results[name]["peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        del experts
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nrelative L2 error vs the fp32 reference, {T} tokens, production dims")
    print(
        f"{'':10s}" + "".join(f"{n:>18s}" for n in results) + f"{'bf16 rounding':>18s}"
    )
    for k in TENSORS:
        rounding = _rel(reference[k].bfloat16().float(), reference[k])
        print(
            f"{k:10s}"
            + "".join(f"{_rel(r[k], reference[k]):18.2e}" for r in results.values())
            + f"{rounding:18.2e}"
        )
    print(
        "sonic recompute vs deepgemm: "
        + " ".join(
            f"{k} {_rel(results['sonic recompute'][k], results['deepgemm'][k]):.2e}"
            for k in TENSORS
        )
    )
    print(
        "held for the backward (MiB): "
        + ", ".join(f"{n} {r['held_MiB']:.0f}" for n, r in results.items())
        + "; peak (GiB): "
        + ", ".join(f"{n} {r['peak_GiB']:.1f}" for n, r in results.items())
    )
    return reference, results


@needs_sonic
def test_sonicmoe_experts_match_deepgemm_and_fp32(runs):
    """Output and every gradient, both memory modes: the bf16 level of an fp32
    reference, and as close to it as the DeepGEMM path (the bound: 1.5x its
    error plus 2e-3; measured the two within a few percent of each other)."""
    reference, results = runs
    for name in ("sonic recompute", "sonic store"):
        for k in TENSORS:
            err = _rel(results[name][k], reference[k])
            bound = 1.5 * _rel(results["deepgemm"][k], reference[k]) + 2e-3
            assert err < bound and err < 1e-2, (name, k, err, bound)
        for k in TENSORS:
            assert _rel(results[name][k], results["deepgemm"][k]) < 1e-2, (name, k)


@needs_sonic
def test_memory_modes(runs):
    """ "store" holds SonicMoE's pre-activation (T * K * 2F bf16) and its metadata
    for the backward, "recompute" nothing of the experts; same numbers."""
    _, results = runs
    h_mib = T * K * 2 * F * 2 / 2**20
    held = {n: r["held_MiB"] for n, r in results.items()}
    assert held["sonic store"] - held["sonic recompute"] > 0.9 * h_mib, held
    assert held["sonic recompute"] < 0.1 * h_mib, held
    for k in TENSORS:
        assert _rel(results["sonic store"][k], results["sonic recompute"][k]) < 1e-6, k


def _layer(expert_gemm, sonic_state="recompute"):
    paths = dict(token_dispatch="gather", expert_gemm=expert_gemm)
    if expert_gemm == "sonicmoe":
        paths["sonic_state"] = sonic_state
    moe = _build(_flavor(**paths).layers[1].feed_forward)
    # one random draw of every parameter, in the same order for both backends
    generator = torch.Generator(device="cuda").manual_seed(7)
    for name, p in moe.named_parameters():
        _fill(p, generator, scale=0.05 if name.startswith("router") else 0.02)
    return moe


def _run_layer(moe):
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(T, D, generator=g, device="cuda").bfloat16().requires_grad_(True)
    cotangent = torch.randn(T, D, generator=g, device="cuda").bfloat16()
    block = _block(moe, "feed_forward")
    y = block(x)
    y.backward(cotangent)
    grads = {n: _cpu(p.grad) for n, p in moe.named_parameters()}
    return dict(out=_cpu(y), dx=_cpu(x.grad), **grads)


@needs_sonic
def test_whole_layer_with_router_and_shared_expert_matches_deepgemm():
    """OptMoE's own router (scores, experts, statistics) and shared expert around
    the backend: the output, the input gradient and every parameter gradient
    (the router's through the scores) agree between the two backends."""
    pytest.importorskip("deep_gemm")
    results = {}
    for name in ("deepgemm", "sonicmoe"):
        moe = _layer(name)
        results[name] = _run_layer(moe)
        del moe
        gc.collect()
        torch.cuda.empty_cache()
    worst = {
        k: _rel(results["sonicmoe"][k], results["deepgemm"][k])
        for k in results["deepgemm"]
    }
    print(
        "\nwhole layer, sonicmoe vs deepgemm (relative L2):",
        {k: f"{v:.1e}" for k, v in worst.items()},
    )
    for k, v in worst.items():
        assert v < 2e-2, (k, v)
    assert any("router.gate" in k for k in worst) and any(
        "shared_experts" in k for k in worst
    )


@needs_sonic
@pytest.mark.parametrize("sonic_state", ["recompute", "store"])
def test_compiled_block_runs_sonicmoe_outside_the_compiler(sonic_state):
    """Under compile_blocks' torch.compile(fullgraph=True) in a rematerialisation
    checkpoint: SonicMoE's kernels run eagerly at runtime (never traced: its own
    asserts against compiling hold), the compiled block equals the eager one, and
    the replay runs the up projection again but not the down projection."""
    from torchtitan.models.opt_moe import sonicmoe_experts

    tokens, num_experts, top_k = 1024, 16, 4
    config = (
        build_model_config(
            "mis-8b-proxy",
            parallelism=ParallelismConfig(),
            seq_len=512,
            n_layers=1,
            swa_pattern="F",
            rope_pattern=None,
            moe_hidden_dim=128,
            num_experts=num_experts,
            top_k=top_k,
            token_dispatch="gather",
            expert_gemm="sonicmoe",
            sonic_state=sonic_state,
        )
        .layers[0]
        .feed_forward.routed_experts
    )
    experts = _build(config)
    dim = experts.w13.in_features
    generator = torch.Generator(device="cuda").manual_seed(5)
    _fill(experts.w13.weight, generator)
    _fill(experts.w2.weight, generator)
    g = torch.Generator(device="cuda").manual_seed(1)
    x0 = torch.randn(tokens, dim, generator=g, device="cuda").bfloat16()
    ids = (
        torch.rand(tokens, num_experts, generator=g, device="cuda")
        .topk(top_k, dim=-1)
        .indices
    )
    scores0 = torch.softmax(
        torch.randn(tokens, top_k, generator=g, device="cuda"), dim=-1
    )
    cotangent = torch.randn(tokens, dim, generator=g, device="cuda").bfloat16()
    rmap = torch.zeros(tokens, num_experts, dtype=torch.bool, device="cuda").scatter_(
        1, ids, True
    )

    sonic = sonicmoe_experts._sonic()
    calls = {"up": 0, "down": 0}

    def counted(name, original):
        def wrapper(*args, **kwargs):
            assert not torch.compiler.is_compiling()
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    originals = (sonic.up_forward, sonic.down_forward)
    sonic.up_forward = counted("up", originals[0])
    sonic.down_forward = counted("down", originals[1])
    out = {}
    try:
        # the dynamo settings the model sets for compile_blocks
        with torch._dynamo.config.patch(
            capture_scalar_outputs=True,
            skip_fwd_side_effects_in_bwd_under_checkpoint=True,
            fail_on_recompile_limit_hit=True,
        ):
            for mode in ("eager", "compiled"):
                block = _block(experts, compile=mode == "compiled")
                x = x0.clone().requires_grad_(True)
                scores = scores0.clone().requires_grad_(True)
                experts.zero_grad(set_to_none=True)
                calls.update(up=0, down=0)
                y = block(x, scores, ids, rmap)
                y.backward(cotangent)
                out[mode] = dict(
                    out=_cpu(y),
                    dx=_cpu(x.grad),
                    dscores=_cpu(scores.grad),
                    dw13=_cpu(experts.w13.weight.grad),
                    dw2=_cpu(experts.w2.weight.grad),
                    calls=dict(calls),
                )
                torch._dynamo.reset()
    finally:
        sonic.up_forward, sonic.down_forward = originals
    print(
        f"\n{sonic_state}: kernel launches per step (forward + replay)",
        {m: o["calls"] for m, o in out.items()},
    )
    for k in ("out", "dx", "dscores", "dw13", "dw2"):
        assert _rel(out["compiled"][k], out["eager"][k]) < 1e-5, k
    replays = 1 if sonic_state == "recompute" else 0
    assert out["compiled"]["calls"] == {"up": 1 + replays, "down": 1}, out["compiled"][
        "calls"
    ]
    assert out["eager"]["calls"] == {"up": 1 + replays, "down": 1 + replays}, out[
        "eager"
    ]["calls"]


def _flags(**kw):
    config = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=512,
        n_layers=4,
        swa_pattern="SSSF",
        rope_pattern=None,
        n_dense_layers=1,
        dense_hidden_dim=256,
        expert_gemm="sonicmoe",
        **{"moe_hidden_dim": 128, **kw},
    )
    return [layer.feed_forward.routed_experts.recompute for layer in config.layers[1:]]


def test_memory_mode_per_layer_and_build_errors():
    """The first sonic_store_layers MoE layers (after the dense ones) store, the
    others recompute; unsupported combinations raise at config time."""
    assert _flags() == [True, True, True]
    assert _flags(sonic_store_layers=2) == [False, False, True]
    assert _flags(sonic_state="store") == [False, False, False]
    for bad in (
        dict(token_dispatch="scatter_add"),
        dict(sonic_store_layers=-1),
        dict(sonic_state="store", sonic_store_layers=1),
        dict(moe_hidden_dim=100),
    ):
        with pytest.raises(ValueError):
            _flags(**bad)
    with pytest.raises(ValueError, match="need expert_gemm"):
        build_model_config(
            "mis-8b-proxy",
            parallelism=ParallelismConfig(),
            seq_len=512,
            n_layers=2,
            swa_pattern="SF",
            rope_pattern=None,
            sonic_state="store",
        )
