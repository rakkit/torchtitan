# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepEP custom ops must be traceable by torch.compile.

`deepep::dispatch` and `deepep::combine` are opaque `torch.library` ops. Without a
fake (meta) impl they raise

    NotImplementedError: deepep::dispatch: attempted to run this operator with Meta
    tensors, but there was no fake impl or Meta kernel registered.

the moment any enclosing module is compiled, so `moe_comm_backend="deepep"` could only
run with model compile disabled. Unlike hybridep -- whose dispatch
outputs are statically sized -- DeepEP's training layout is COMPACT/deduplicated: the
received-token count only exists after a host sync, so the fake must produce an unbacked
symint and every downstream consumer must stay symbolic.

These tests are CPU-only in spirit (nothing is executed, only traced through
FakeTensorMode), but importing `deep_ep` needs CUDA_HOME and the package, so they skip
cleanly where it is unavailable.
"""

import pytest
import torch

# NOT importorskip: it only catches ImportError, and deep_ep/__init__.py raises a bare
# AssertionError from find_cuda_home() when CUDA_HOME is unset, so the whole module
# aborts with a traceback instead of skipping.
try:
    from torchtitan.distributed.deepep import deepep
except Exception as _e:  # noqa: BLE001
    pytest.skip(
        f"deep_ep not importable (needs the package and CUDA_HOME): {_e!r}",
        allow_module_level=True,
    )

T, H, K, NUM_EXPERTS, NUM_LOCAL_EXPERTS = 64, 32, 2, 8, 2


def _fake_mode():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return FakeTensorMode(shape_env=ShapeEnv())


def _inputs():
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(-1, NUM_LOCAL_EXPERTS, (T, K), device="cuda", dtype=torch.long)
    scores = torch.rand(T, K, device="cuda", dtype=torch.float32)
    return x, idx, scores


def _dispatch(x, idx, scores, cudagraphable=False):
    return torch.ops.deepep.dispatch(
        x,
        idx,
        scores,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL_EXPERTS,
        num_tokens_per_rank=T,
        cudagraphable=cudagraphable,
    )


def test_dispatch_fake_shapes_and_devices():
    """The compact fake must produce an unbacked row count and a CPU count tensor."""
    with _fake_mode():
        x, idx, scores = _inputs()
        recv_x, recv_idx, recv_scores, counts, handle_id = _dispatch(x, idx, scores)

        # Hidden dim, dtypes and the trailing top-k dim are all static.
        assert recv_x.shape[1] == H
        assert recv_x.dtype == x.dtype
        assert recv_idx.shape[1] == K and recv_idx.dtype == torch.long
        assert recv_scores.shape[1] == K and recv_scores.dtype == torch.float32

        # The received-token count is data-dependent: it must NOT specialize to a
        # constant, or the traced graph would silently bake in one step's routing.
        assert isinstance(recv_x.shape[0], torch.SymInt)
        # ...and it must be the SAME symbol across all three per-token outputs.
        assert str(recv_idx.shape[0]) == str(recv_x.shape[0])
        assert str(recv_scores.shape[0]) == str(recv_x.shape[0])

        # Per-local-expert counts come back on CPU (the real impl builds them from the
        # host-synced list), sized by num_local_experts.
        assert counts.device.type == "cpu"
        assert counts.shape == (NUM_LOCAL_EXPERTS,)
        assert counts.dtype == torch.int32

        assert handle_id.device.type == "cpu"
        assert handle_id.dtype == torch.int64


def test_expand_layout_fake_refuses_rather_than_guesses():
    """cudagraphable=True must raise, not invent a row count.

    The expand layout's size is computed inside the DeepEP kernel. Guessing it would
    mis-size the traced graph silently, which is strictly worse than not compiling.
    """
    with _fake_mode():
        x, idx, scores = _inputs()
        with pytest.raises(Exception, match="expand"):
            _dispatch(x, idx, scores, cudagraphable=True)


def test_full_compact_chain_traces_to_the_original_token_count():
    """dispatch -> permute -> expert compute -> unpermute -> combine.

    This is the whole training path. It is the regression test for two specific
    specialization traps:
      * `_permute_tokens` used `len(hidden_states)`, which coerces an unbacked symint to
        a Python int and raises GuardOnDataDependentSymNode;
      * `combine` cannot recover the caller's original token count from its
        (permuted, deduplicated) input, so it takes `num_tokens` explicitly.
    """
    with _fake_mode():
        x, idx, scores = _inputs()
        recv_x, recv_idx, recv_scores, counts, handle_id = _dispatch(x, idx, scores)

        # The grouped-GEMM path moves the counts to the device.
        tokens_per_expert = counts.to(recv_x.device)
        assert tokens_per_expert.device.type == "cuda"

        permuted, _permuted_scores, permuted_indices = deepep._permute_tokens(
            recv_x, recv_idx, recv_scores
        )
        expert_out = permuted * 2.0  # stand-in for the grouped GEMM
        unpermuted = deepep._unpermute_tokens(
            expert_out, permuted_indices, recv_x.shape[0]
        )
        combined = torch.ops.deepep.combine(unpermuted, handle_id, T, True)

    # Combine reduces back to the caller's token count -- static, not unbacked.
    assert combined.shape == (T, H)
    assert combined.dtype == x.dtype


def test_op_schemas_carry_the_arguments_the_fakes_need():
    """The fakes cannot be written without these two arguments; pin them.

    `num_local_experts` sizes the compact count tensor and `num_tokens` sizes combine's
    output. Both were absent from the original schemas (and from 0.4.0's), which is why
    no fake impl was possible.
    """
    dispatch_schema = str(torch._C._jit_get_schemas_for_operator("deepep::dispatch")[0])
    combine_schema = str(torch._C._jit_get_schemas_for_operator("deepep::combine")[0])
    assert "int num_local_experts" in dispatch_schema
    assert "int num_tokens" in combine_schema


def test_importing_deepep_does_not_touch_global_dynamo_config():
    """The contract is the opposite of what this test once asserted.

    An earlier version set `capture_dynamic_output_shape_ops = True` at module scope so
    dynamo would trace these ops instead of graph-breaking on them. That made things
    worse, not better: dynamo then reached `_handle_cache[handle_id.item()]`, where under
    tracing `.item()` is an unbacked SymInt and cannot be a dict key, turning a working
    graph break into a hard failure. 0.4.0 registered no fakes and never set the flag, so
    it graph-broke -- which is why the 100B EP=4 proposal runs trained fine with
    `compile.enable = True`.

    It is also a GLOBAL setting, so flipping it on import changed how every
    dynamic-output-shape op in the model was handled, not just these two.
    """
    assert torch._dynamo.config.capture_dynamic_output_shape_ops is False


def test_dispatch_is_tagged_dynamic_output_shape():
    """The op's output shape is genuinely data-dependent, so it must say so.

    NOTE this tag is NOT what unblocks compile -- an earlier version of this docstring
    claimed it was, which log section 37 retracts. The unhashable-SymInt failure comes
    from the handle lookup in setup_context, not from the fake's output.
    """
    assert torch.Tag.dynamic_output_shape in torch.ops.deepep.dispatch.default.tags
    # combine's output is statically sized; it must NOT carry the tag.
    assert torch.Tag.dynamic_output_shape not in torch.ops.deepep.combine.default.tags
