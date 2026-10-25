# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The residual capture (OPTMoEModel._trunk, armed on logging steps) under the
production block compile (fullgraph=True) and SelectiveAC, on one GPU: the
blocks trace without a break, arming never recompiles, the captured RMS equals
an eager recomputation, and an armed step's forward and backward are bitwise
those of an unarmed step (test 8, the hard check).
"""

from __future__ import annotations

import math

import pytest
import torch

from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.opt_moe import model_registry
from torchtitan.optimizers import model_stats

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DOCS = (640, 384)  # one block-causal batch of 1024 tokens, two documents
VOCAB = 1024


def _model(
    *, dtype: torch.dtype, seed: int = 0, ac: bool = True, compile_: bool = True
):
    cfg = model_registry("gauge-d16-w512")
    cfg.n_layers, cfg.vocab_size = 2, VOCAB
    cfg.layer.norm_type = cfg.layer.attention.norm_type = cfg.norm_type = "rmsnorm"
    cfg.layer.depth_init = "identity"
    cfg._expand_layers()
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(seed)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cuda"))
    model.to(dtype)
    if ac:
        SelectiveAC.Config().build().apply(model)
    if compile_:
        apply_compile(
            model,
            compile_config=CompileConfig(
                components=["model", "loss"], backend="inductor"
            ),
            parallel_dims=ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=1,
                enable_sequence_parallel=False,
            ),
        )
    return model


def _batch(seed: int):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (sum(DOCS),), generator=g).cuda()
    positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
    return tokens, positions


def _masks(model, positions):
    return model.get_attention_masks(
        positions=positions,
        padding_mask=None,
        max_num_documents=None,
        max_context_length=4096,
    )


def _step(model, tokens, positions, *, armed: bool):
    """One forward + backward; returns (output, loss, {name: grad}) and leaves no grads."""
    model.zero_grad(set_to_none=True)
    if armed:
        model_stats.arm_residual_capture([model])
    out = model(tokens, positions, _masks(model, positions))
    loss = torch.nn.functional.cross_entropy(out.float(), torch.roll(tokens, -1))
    loss.backward()
    stats = model_stats.residual_metrics([model], parallel_dims=None) if armed else {}
    assert model.residual_capture is None
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return out.detach().clone(), loss.detach().clone(), grads, stats


def _assert_bitwise(a, b, label):
    out_a, loss_a, grads_a, _ = a
    out_b, loss_b, grads_b, _ = b
    assert torch.equal(out_a, out_b), f"{label}: output"
    assert torch.equal(loss_a, loss_b), f"{label}: loss"
    assert grads_a.keys() == grads_b.keys()
    for n in grads_a:
        assert torch.equal(grads_a[n], grads_b[n]), f"{label}: grad {n}"


def test_block_traces_to_one_graph_without_breaks():
    """Test 5: torch._dynamo.explain over the AC-wrapped block (unchanged code)."""
    torch._dynamo.reset()
    model = _model(dtype=torch.bfloat16, compile_=False)
    tokens, positions = _batch(0)
    with torch.no_grad():
        x = model.tok_embeddings(tokens)
    explain = torch._dynamo.explain(model.layers["0"])(
        x, _masks(model, positions), positions
    )
    assert explain.graph_break_count == 0, explain.break_reasons
    assert explain.graph_count == 1, explain


def test_arming_never_recompiles():
    """Test 6: after warmup, armed and unarmed steps run on the same compiled
    graphs with no recompile."""
    torch._dynamo.reset()
    model = _model(dtype=torch.bfloat16)
    _step(model, *_batch(0), armed=False)  # warmup compiles
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    with torch._dynamo.config.patch(error_on_recompile=True):
        for i, armed in enumerate((True, False, True, False), start=1):
            _step(model, *_batch(i), armed=armed)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def test_captured_rms_equals_eager_recomputation():
    """Test 7 (fp32): the capture of the compiled, AC-wrapped model matches the RMS
    recomputed by hooks on an eager twin with the same weights and input."""
    torch._dynamo.reset()
    comp = _model(dtype=torch.float32)
    eager = _model(dtype=torch.float32, ac=False, compile_=False)
    tokens, positions = _batch(7)
    seen = {}
    handles = []
    for i, block in eager.layers.items():

        def pre(m, args, kwargs, i=i):
            seen[("in", i)] = args[0].detach().double()

        def post(m, args, kwargs, out, i=i):
            seen[("out", i)] = out.detach().double()

        handles += [
            block.register_forward_pre_hook(pre, with_kwargs=True),
            block.register_forward_hook(post, with_kwargs=True),
        ]
    with torch.no_grad():
        eager(tokens, positions, _masks(eager, positions))
    for h in handles:
        h.remove()
    last = max(eager.layers.keys(), key=int)
    want = {}
    for i in eager.layers.keys():
        x, y = seen[("in", i)], seen[("out", i)]
        want[f"residual_rms/in/L-{i}"] = math.sqrt(float(x.square().mean()))
        want[f"residual_rms/write/L-{i}"] = math.sqrt(float((y - x).square().mean()))
    want["residual_rms/final_norm_in"] = math.sqrt(
        float(seen[("out", last)].square().mean())
    )
    _, _, _, got = _step(comp, tokens, positions, armed=True)
    assert got.keys() == want.keys()
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-5), k


def test_armed_forward_and_backward_are_bitwise():
    """Test 8, the hard check (bf16, SelectiveAC, compiled): unarmed / armed /
    unarmed on one model, and a fresh model compiled during an armed step
    (armed first) -- outputs, loss and every gradient torch.equal."""
    torch._dynamo.reset()
    a = _model(dtype=torch.bfloat16, seed=3)
    tokens, positions = _batch(11)
    _step(a, tokens, positions, armed=False)  # compile
    u1 = _step(a, tokens, positions, armed=False)
    on = _step(a, tokens, positions, armed=True)
    u2 = _step(a, tokens, positions, armed=False)
    assert on[3], "the armed step captured nothing"
    _assert_bitwise(u1, on, "unarmed vs armed")
    _assert_bitwise(u1, u2, "unarmed vs unarmed after an armed step")

    torch._dynamo.reset()
    b = _model(dtype=torch.bfloat16, seed=3)
    first = _step(b, tokens, positions, armed=True)  # compiled while armed
    then = _step(b, tokens, positions, armed=False)
    _assert_bitwise(first, then, "fresh model: armed first vs unarmed after")
    # Across the two models this also compares two compilations of the same code.
    _assert_bitwise(u1, first, "first model unarmed vs fresh model armed first")
