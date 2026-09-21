# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepEP v2 primitives for MoE Expert Parallel, on the unified ``ElasticBuffer`` API.

DeepEP v2 (>= 2.0.0) collapses the v1 two-path design -- high-throughput (HT,
``buffer.dispatch``/``combine``) and low-latency (LL,
``buffer.low_latency_dispatch``/``combine``) -- into a SINGLE ``dispatch``/``combine``
on ``deep_ep.ElasticBuffer``. There is one buffer, one pair of custom ops, and one
``DispatchState`` for both modes; only ``dispatch`` branches. The branch is chosen at
runtime by the GRAD context, not by prefill-vs-decode: ``dispatch_tokens`` forces the
compact path whenever ``torch.is_grad_enabled()`` (training), so the expand path is taken
only by an inference forward (no grad) that also set ``cuda_graph_compatible=True`` -- which covers
BOTH prefill and decode. Combine is handle-driven and mode-agnostic.

- training (``cuda_graph_compatible=False``, the default; also forced under autograd): ``do_expand=False`` +
  ``do_cpu_sync=True`` -- a compact, deduplicated layout. ``_permute_tokens`` gathers it
  into expert-major order using ``handle.num_recv_tokens_per_expert_list`` for the grouped
  GEMM. Full autograd is provided by the custom ops below (dispatch backward is a combine,
  combine backward is a dispatch). The total received count is data-dependent and needs a
  host sync, so this path is NOT CUDA-graph-compatible.
- inference -- BOTH prefill and decode (``cuda_graph_compatible=True``, under no_grad): ``do_expand=True`` +
  ``do_cpu_sync=False`` -- the static "one-token-per-expert-slot" expanding layout,
  routing-independent (correct even as gating changes between captured replays) and with no
  host sync, so the MoE forward is CUDA-graph-capturable. Per-expert offsets come from the
  device-side ``handle.psum_num_recv_tokens_per_expert`` (no CPU sync). Inference-only: the
  expanding layout "must not be backward" per the DeepEP kernels.

Routing scores are applied to expert outputs in plain PyTorch (in ``combine_tokens``,
before the pure-reduction combine op), so autograd handles the score gradient, the custom
ops stay pure communication, and combine works unchanged in both modes (``combine``
ignores ``topk_weights`` in expand mode anyway).
"""

from dataclasses import dataclass


import torch

from torch.distributed import ProcessGroup

try:
    from deep_ep import ElasticBuffer
except ImportError as e:
    raise ImportError(
        "DeepEP v2 (>= 2.0.0, ElasticBuffer) is required for this module. "
        "Install from: https://github.com/deepseek-ai/DeepEP"
    ) from e


# Global buffer (single buffer per process, recreated if the group changes or a
# larger size is needed). v2 uses ONE ElasticBuffer for both training and inference.
_buffer: ElasticBuffer | None = None

# Global cache for dispatch handles (EPHandle objects), keyed by an int handle_id.
# The torch.library custom ops can only pass tensors across the op boundary, so we
# smuggle the opaque EPHandle through a CPU int64 handle_id tensor + this cache.
# SAC saves the handle_id tensor; we use it to retrieve the non-tensor handle.
_handle_cache: dict = {}
_handle_counter: int = 0

# Pending combine event for deferred synchronization. The caller MUST call
# sync_combine() before using the result. Process-local + single-threaded, so a
# module var suffices.
_pending_combine_event = None


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle_id tensor on CPU to avoid a GPU-CPU sync."""
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


# ============================================================================
# Custom Op Registration for SAC Integration + autograd
# ============================================================================
#
# ElasticBuffer.dispatch/combine are not autograd-aware. We wrap them in
# torch.library custom ops so (a) SAC saves the comm outputs instead of recomputing
# them and (b) we attach manual backward: dispatch backward is a combine and combine
# backward is a dispatch (the DeepEP forward/backward duality). The opaque EPHandle
# is passed across the op boundary via a CPU handle_id + _handle_cache.

_lib = torch.library.Library("deepep", "DEF")

# dispatch returns: (recv_x, recv_topk_idx, recv_scores, num_recv_per_expert, handle_id).
# recv_topk_idx is the per-received-token local-expert assignment, used by the compact
# path to gather tokens into expert-major order (it is an empty placeholder in expand mode,
# whose static layout is already expert-grouped).
# ``dynamic_output_shape`` declares what is true of the compact layout: the received-token
# count is host-synced and data-dependent, so the fake impl below returns an unbacked
# symint. This is the same tag nonzero/unique/item carry.
#
# NOTE it is NOT sufficient to make this op compile -- see the block above
# _dispatch_setup_context; see the block above it for how the handle lookup is kept
# out of tracing.
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "int num_experts, int num_local_experts, int num_tokens_per_rank, "
    "bool cuda_graph_compatible) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
    tags=(torch.Tag.dynamic_output_shape,),
)
# combine returns: combined_x. ``will_backward`` is the caller's outer grad state
# (torch.is_grad_enabled() evaluated before the op): it is the only reliable signal for
# whether a backward will consume the cached handle, since inside a custom-op forward
# autograd disables grad regardless of the outer context. When False (generator no_grad /
# inference), the op frees the handle itself (setup_context never runs).
_lib.define(
    "combine(Tensor x, Tensor handle_id, int num_tokens, bool will_backward) -> Tensor"
)
# The backwards are ops in their own right so the opaque-handle lookup happens in a CUDA
# impl AT RUNTIME rather than in setup_context DURING TRACING. Looking it up while tracing
# is what made compile impossible: `_handle_cache[handle_id.item()]` needs `.item()`, which
# under tracing is an unbacked SymInt and cannot be a dict key
# ("TypeError: unhashable type: non-nested SymInt").
#
# `like_x` / `like_recv` are shape carriers: each backward's output has the shape of a
# tensor the forward already produced, so the fakes below derive their sizes from an input
# instead of minting a new unbacked size. hybridep and minimal_async_ep are built the same
# way, and both compile.
_lib.define(
    "dispatch_backward(Tensor grad_recv_x, Tensor grad_recv_scores, Tensor handle_id, "
    "Tensor like_x, Tensor like_scores) -> (Tensor, Tensor)"
)
_lib.define(
    "combine_backward(Tensor grad_combined, Tensor handle_id, Tensor like_recv) -> Tensor"
)


# Fallback dispatch/combine SM count when deep_ep's bandwidth heuristic cannot run
# (see _resolve_dispatch_num_sms). num_sms only affects performance, not correctness;
# 20 matches vLLM's deep_ep integration (all2all.py uses num_sms=20).
_DEEPEP_MULTINODE_NUM_SMS = 20


def _resolve_dispatch_num_sms(buffer, num_experts: int, num_topk: int) -> int:
    """SM count for the dispatch kernel (also reused by combine via the handle).

    deep_ep's get_theoretical_num_sms() derives the count from link bandwidths,
    but its RDMA-bandwidth auto-detect can report 0 GB/s on some multi-node
    topologies, making the heuristic divide by zero. On that failure, fall back
    to a fixed count (_DEEPEP_MULTINODE_NUM_SMS); num_sms only affects
    performance, not correctness. dispatch() stores the value on the returned
    handle and combine() reuses it.
    """
    try:
        return buffer.get_theoretical_num_sms(num_experts, num_topk)
    except ZeroDivisionError:
        num_device_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        return min(_DEEPEP_MULTINODE_NUM_SMS, num_device_sms)


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    num_tokens_per_rank: int,
    cuda_graph_compatible: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute DeepEP v2 dispatch.

    ``cuda_graph_compatible=False`` (training / any grad-enabled forward): the COMPACT (non-expand)
    layout. ``recv_x`` is
    DEDUPLICATED -- one row per unique received token -- while ``recv_topk_idx`` gives each
    token's local-expert assignments (-1 for picks not on this rank). ``dispatch_tokens``
    gathers this into expert-major order for the grouped GEMM (matching the v1 path). A host
    sync gives exact per-expert counts, so it is NOT CUDA-graph-compatible.
    ``cuda_graph_compatible=True`` (inference, both prefill and decode): the static ``do_expand`` layout, already
    expert-grouped (tokens packed in ``[0:sum(counts)]``, tail unused) with no host sync, so
    the forward is CUDA-graph-capturable; per-expert counts come from the device-side
    ``psum_num_recv_tokens_per_expert``.
    """
    global _buffer
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    # num_local_experts is carried only so the fake impl can size num_recv_per_expert;
    # the real counts come from the handle.
    del num_local_experts

    # Resolve num_sms ourselves and pass it explicitly: the resolver calls deep_ep's
    # bandwidth heuristic but catches its multi-node RDMA-bandwidth divide-by-zero, so
    # dispatch() never falls into that heuristic internally with the default num_sms=0.
    # See _resolve_dispatch_num_sms.
    num_sms = _resolve_dispatch_num_sms(buffer, num_experts, topk_idx.shape[1])
    recv_x, recv_topk_idx, recv_scores, handle, _event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        # Denote C = num_tokens_per_rank. DeepEP uses C as the per-rank layout stride
        # (global token ID = rank * C + local token ID) and, in expand mode,
        # to size recv_x. MoE physically pads x so C is identical across ranks.
        num_max_tokens_per_rank=num_tokens_per_rank,
        num_sms=num_sms,
        do_expand=cuda_graph_compatible,
        do_cpu_sync=not cuda_graph_compatible,
    )

    handle_id = _get_next_handle_id()
    _handle_cache[handle_id.item()] = handle

    # Per-local-expert received-token counts for the grouped GEMM.
    if cuda_graph_compatible:
        # Expand mode: no host sync allowed. Recover per-expert counts from the
        # device-side inclusive prefix sum (expert_alignment defaults to 1, so this is
        # a plain prefix sum). GroupedExperts.forward cumsums these back into grouped-mm offs.
        psum = handle.psum_num_recv_tokens_per_expert
        num_recv_per_expert = torch.diff(psum, prepend=psum.new_zeros(1)).to(
            torch.int32
        )
        # Expand layout is already expert-grouped; no gather needed -> empty placeholder
        # (a torch.library op must return a Tensor, not None).
        recv_topk_idx = recv_x.new_empty(0, dtype=torch.long)
    else:
        # Compact mode: exact counts are a CPU list (available after the host sync).
        num_recv_per_expert = torch.tensor(
            handle.num_recv_tokens_per_expert_list, dtype=torch.int32, device="cpu"
        )
    return recv_x, recv_topk_idx, recv_scores, num_recv_per_expert, handle_id


# ---------------------------------------------------------------------------
# HOW THIS BACKEND COMPILES
#
# The opaque EPHandle cannot be a graph value, so dispatch returns an integer
# ``handle_id`` tensor and the handle itself lives in a module-level dict, ``_handle_cache``.
# Resolving it needs ``handle_id.item()``, and WHERE that happens decides whether the
# backend compiles:
#
#   setup_context  -> runs DURING TRACING. There ``.item()`` is an unbacked SymInt, and a
#                     SymInt cannot be a dict key:
#                         File "torch/__init__.py", in __hash__
#                             raise TypeError("unhashable type: non-nested SymInt")
#                     dynamo surfaces this as a misleading "RuntimeError when making fake
#                     tensor call ..." naming deepep.dispatch with every argument shown as
#                     concrete, which is why it took three attempts to find.
#   a CUDA impl    -> runs AT RUNTIME on a real tensor, where ``.item()`` is a real int.
#
# So the backwards are custom ops of their own (``dispatch_backward`` / ``combine_backward``)
# and the setup_contexts save only TENSORS: the handle_id plus shape carriers for the
# outputs. Fakes for all four ops let dynamo trace the whole region, forward and backward.
# hybridep.py and minimal_async_ep/api.py are built the same way and compile for the same
# reason.
#
# Verified under FakeTensorMode(shape_env=ShapeEnv()) with requires_grad=True inputs -- the
# login-node repro that finds this class of bug in seconds instead of an sbatch cycle --
# both with capture_dynamic_output_shape_ops left False (dynamo graph-breaks at the
# data-dependent dispatch; deepep runs eager, everything around it compiles) and True
# (dynamo traces straight through). Both reach backward and produce correctly shaped grads.
# ---------------------------------------------------------------------------


def _dispatch_setup_context(ctx, inputs, output):
    # NO .item() here. This runs during tracing, where handle_id.item() is an unbacked
    # SymInt and cannot key a dict. The tensor is saved instead and resolved inside the
    # backward OP's CUDA impl, which only ever runs eagerly.
    x, _topk_idx, topk_weights, *_ = inputs
    *_, handle_id = output
    ctx.input_dtype = x.dtype
    ctx.save_for_backward(handle_id, x, topk_weights)


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_topk_idx,
    grad_recv_scores,
    grad_num_recv,
    grad_handle_id,
):
    """Backward for dispatch: a combine of the gradients.

    recv_topk_idx is non-differentiable, so grad_recv_topk_idx is ignored.
    """
    handle_id, like_x, like_scores = ctx.saved_tensors
    if grad_recv_x is None:
        # The op below is what normally frees the handle; skipping it would leak.
        # Guarded because .item() is only legal outside tracing -- and under
        # AOTAutograd grad_recv_x is never None, so this branch is eager-only.
        if not torch.compiler.is_compiling():
            _handle_cache.pop(handle_id.item(), None)
        return None, None, None, None, None, None, None

    if grad_recv_scores is None:
        grad_recv_scores = torch.zeros_like(like_scores)
    grad_x, grad_scores = torch.ops.deepep.dispatch_backward(
        grad_recv_x, grad_recv_scores, handle_id, like_x, like_scores
    )
    # Order matches op inputs: x, topk_idx, topk_weights, num_experts,
    # num_local_experts, num_tokens_per_rank, cuda_graph_compatible.
    return (
        grad_x.to(ctx.input_dtype),
        None,
        grad_scores.to(ctx.input_dtype),
        None,
        None,
        None,
        None,
    )


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor, handle_id: torch.Tensor, num_tokens: int, will_backward: bool
) -> torch.Tensor:
    """Execute DeepEP v2 combine (pure reduction; scores already applied upstream)."""
    global _buffer, _pending_combine_event
    # num_tokens is carried only so the fake impl can size the output; the real
    # reduction length comes from the handle.
    del num_tokens
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"

    # When no backward will run (generator forward under no_grad/inference_mode), the
    # dispatch setup_context never fires to free the handle, so pop it here. When a backward
    # will run (training), keep it: _combine_setup_context pops it for combine-backward.
    # ``will_backward`` is the caller's OUTER grad state -- inside this forward impl
    # torch.is_grad_enabled() is always False (autograd disables grad during forward), so it
    # cannot tell training from inference here.
    if not will_backward:
        handle = _handle_cache.pop(handle_id.item(), None)
    else:
        handle = _handle_cache.get(handle_id.item())
    assert handle is not None, f"Handle not found for handle_id={handle_id.item()}"

    combined, _combined_weights, after_event = buffer.combine(
        x,
        handle=handle,
        topk_weights=None,
        async_with_compute_stream=True,
    )
    # DeepEP's contract for async_with_compute_stream=True (their docstring: "the
    # current stream will not wait for the communication kernels to be finished if
    # set", "event ... valid only if async_with_compute_stream is set") is that the
    # CALLER must wait on the returned event before consuming `combined`.
    #
    # That wait used to live only in sync_combine(), which returns early under
    # torch.compile because CUDA event ops are not traceable. Once dynamo started
    # tracing THROUGH deepep (it graph-broke before the fakes existed, which is why
    # 0.4.0 was unaffected), the wait silently disappeared and a compiled graph read
    # `combined` while the comm kernel was still in flight -> non-finite loss at
    # step 2.
    #
    # Waiting inside the op honours the contract in BOTH paths, because a custom-op
    # impl always runs eagerly at runtime. It does NOT disable DeepEP's overlap:
    # async_with_compute_stream stays True, the kernels still launch on the comm
    # stream, and work already queued on the compute stream still overlaps. Eager
    # stream order is unchanged too -- the sole caller (token_dispatcher.combine)
    # calls sync_combine() on the very next statement, so nothing was ever enqueued
    # between the two points.
    if after_event is not None:
        after_event.current_stream_wait()
    _pending_combine_event = None
    return combined


def _combine_setup_context(ctx, inputs, output):
    # As above: save the tensor, resolve it inside the backward op at runtime.
    x, handle_id, _num_tokens, _will_backward = inputs
    ctx.save_for_backward(handle_id, x)


def _combine_backward(ctx, grad_combined):
    """Backward for combine: a dispatch of the gradient (reuses the cached handle).

    Returns grads for op inputs (x, handle_id, num_tokens, will_backward); only x is
    differentiable.
    """
    handle_id, like_recv = ctx.saved_tensors
    grad_x = torch.ops.deepep.combine_backward(grad_combined, handle_id, like_recv)
    return grad_x, None, None, None


# ---------------------------------------------------------------------------
# Fake (meta) impls -- required for torch.compile / AOTAutograd tracing.
#
# Unlike hybridep and minimal_async_ep, whose dispatch outputs are STATICALLY sized
# (x.shape[0] / receive_capacity), the compact training layout is deduplicated: the
# received-token count is a host-synced, data-dependent quantity. It is therefore an
# unbacked symint (``new_dynamic_size()``). See the note below on why that means dynamo
# graph-breaks here, and why that is the correct outcome rather than a problem to fix.
# ---------------------------------------------------------------------------


# DO NOT set torch._dynamo.config.capture_dynamic_output_shape_ops here.
#
# These fakes return an unbacked (data-dependent) row count, so dynamo cannot trace the
# ops and GRAPH-BREAKS on them: the deepep region runs eager while everything around it
# still compiles. The model compile path is not fullgraph, so that break is legal, and it
# is what 0.4.0 did -- which is why the 100B EP=4 proposal runs worked with
# `compile.enable = True` (launch_scripts/2026-proposal/100b-32-nodes.sbatch).
#
# Setting the flag makes dynamo trace through instead of breaking. It then reaches
# `_handle_cache[handle_id.item()]`, where under tracing `.item()` is an unbacked SymInt
# and cannot be a dict key -- "TypeError: unhashable type: non-nested SymInt". So the flag
# converts a working graph break into a hard failure, and being global it would change how
# every dynamic-output-shape op in the model is handled, not just these two.
#
# The real fix is to promote `dispatch_backward` / `combine_backward` to custom ops with
# their own fakes, so the cache lookup happens in a CUDA impl at runtime. Both sibling
# backends are built that way and do compile: `hybridep.py` passes the handle as a
# first-class op value; `minimal_async_ep/api.py` uses save_for_backward.


@torch.library.register_fake("deepep::dispatch")
def _dispatch_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    num_tokens_per_rank: int,
    cuda_graph_compatible: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cuda_graph_compatible:
        # The expand layout's row count is computed inside the DeepEP kernel and is not
        # reproducible here; guessing it would silently mis-size the traced graph. That
        # path is inference-only (no_grad) and is not compiled today, so refuse loudly.
        raise NotImplementedError(
            "deepep::dispatch has no fake impl for the expand (cuda_graph_compatible=True) "
            "layout; run the generator eagerly or use cuda_graph_compatible=False."
        )

    ctx = torch.library.get_ctx()
    # One row per UNIQUE received token: bounded above by num_tokens_per_rank * ep_size,
    # but the exact value needs the host sync, so it is unbacked.
    num_recv = ctx.new_dynamic_size()
    topk = topk_idx.shape[1]

    recv_x = x.new_empty(num_recv, x.shape[1])
    recv_topk_idx = topk_idx.new_empty(num_recv, topk)
    recv_scores = topk_weights.new_empty(num_recv, topk)
    # Compact mode returns the per-local-expert counts on CPU (they come from the
    # host sync); dispatch_tokens immediately moves them to the device.
    num_recv_per_expert = torch.empty(
        num_local_experts, dtype=torch.int32, device="cpu"
    )
    handle_id = torch.empty(1, dtype=torch.int64, device="cpu")
    return recv_x, recv_topk_idx, recv_scores, num_recv_per_expert, handle_id


@torch.library.register_fake("deepep::combine")
def _combine_fake(
    x: torch.Tensor, handle_id: torch.Tensor, num_tokens: int, will_backward: bool
) -> torch.Tensor:
    """Combine reduces back to the caller's original local token count."""
    return x.new_empty(num_tokens, x.shape[1])


# ---------------------------------------------------------------------------
# Backward ops: CUDA impls (the handle lookup lives HERE, at runtime) + fakes.
# ---------------------------------------------------------------------------


@torch.library.impl(_lib, "dispatch_backward", "CUDA")
def _dispatch_backward_impl(
    grad_recv_x: torch.Tensor,
    grad_recv_scores: torch.Tensor,
    handle_id: torch.Tensor,
    like_x: torch.Tensor,
    like_scores: torch.Tensor,
):
    """dispatch's backward IS a combine of the gradients.

    `.item()` is safe here: this runs eagerly on a real CPU tensor, never under tracing.
    That is the entire point of promoting the backward to an op.
    """
    del like_x, like_scores
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before combine"
    # pop: this is the last consumer of the handle in a training step
    # (combine_backward ran first and only read it), so free it here or the
    # cache grows without bound.
    handle = _handle_cache.pop(handle_id.item(), None)
    assert handle is not None, "Handle not found in dispatch backward"
    grad_x, grad_scores, _event = buffer.combine(
        grad_recv_x, handle=handle, topk_weights=grad_recv_scores.float()
    )
    return grad_x, grad_scores


@torch.library.register_fake("deepep::dispatch_backward")
def _dispatch_backward_fake(
    grad_recv_x, grad_recv_scores, handle_id, like_x, like_scores
):
    # Shapes come from the forward's own tensors, so no NEW unbacked size is minted.
    return torch.empty_like(like_x), torch.empty_like(like_scores)


@torch.library.impl(_lib, "combine_backward", "CUDA")
def _combine_backward_impl(grad_combined, handle_id, like_recv):
    """combine's backward IS a dispatch of the gradient, reusing the cached handle."""
    del like_recv
    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    # get, NOT pop: backward runs in reverse, so this fires BEFORE
    # dispatch_backward, which needs the same handle. dispatch_backward is the
    # last consumer and frees it.
    handle = _handle_cache.get(handle_id.item())
    assert handle is not None, "Handle not found in combine backward"
    # num_sms from the handle: with a cached handle, dispatch's automatic
    # get_theoretical_num_sms runs BEFORE num_experts is inferred from it.
    grad_x, _idx, _scores, _handle, _event = buffer.dispatch(
        grad_combined, handle=handle, num_sms=handle.num_sms, do_cpu_sync=False
    )
    return grad_x


@torch.library.register_fake("deepep::combine_backward")
def _combine_backward_fake(grad_combined, handle_id, like_recv):
    return torch.empty_like(like_recv)


torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)


def sync_combine() -> None:
    """Wait the current CUDA stream on the pending async combine.

    Kept for callers that follow DeepEP's documented "wait before you read" contract,
    but the wait now happens inside the combine op itself (see _combine_op_impl), so
    this is normally a no-op. It has to be: under torch.compile this function returns
    early -- CUDA event ops are not traceable -- so it cannot be the only place the
    wait happens. Safe to call multiple times.
    """
    global _pending_combine_event
    if torch.compiler.is_compiling():
        return
    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Bytes for one token's hidden vector (>= 2 so fp8 and bf16 share a buffer)."""
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(
    group: ProcessGroup,
    *,
    hidden: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    use_fp8_dispatch: bool = False,
) -> ElasticBuffer:
    """Get or create the process-global DeepEP v2 ``ElasticBuffer``.

    A single buffer serves both training and inference (v2 unified the HT/LL buffers).
    It is recreated only if the group changes or a larger buffer is needed. The size
    is computed analytically by ``get_buffer_size_hint`` from the MoE settings; v2
    needs ``num_max_tokens_per_rank`` (the max tokens any rank may dispatch in one
    forward) up front because the buffer is sized statically.

    Created with ``explicitly_destroy=True`` so the C++ destructor does NOT auto-run
    ``destroy()`` (-> ``cudaDeviceSynchronize`` + host barrier) on GC: that barrier
    inside a CUDA-graph capture aborts the capture. We never call ``destroy()`` (the
    buffer lives for the process; leaking the comm buffer at exit is fine). Matches
    vLLM's DeepEP buffer usage and the validated v1 low-latency CUDA graph path.
    """
    global _buffer
    needed_bytes = ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank,
        hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    if (
        _buffer is not None
        and _buffer.group == group
        and _buffer.num_bytes >= needed_bytes
    ):
        return _buffer
    _buffer = ElasticBuffer(
        group,
        num_bytes=needed_bytes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        explicitly_destroy=True,
    )
    return _buffer


def _permute_tokens(
    hidden_states: torch.Tensor,
    dispatched_indices: torch.Tensor,
    dispatched_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the compact (deduplicated) dispatch output into expert-major order.

    v2 non-expand ``recv_x`` has one row per unique received token, but a token routed to
    several local experts must appear once per expert for the grouped GEMM. This expands and
    sorts by expert id (a token's valid count comes from ``dispatched_indices != -1``),
    matching the validated v1 high-throughput path so numerics are identical.

    Args:
        hidden_states: Received tokens ``[num_recv_tokens, hidden]`` (deduplicated).
        dispatched_indices: Local expert ids per received token ``[num_recv_tokens, topk]``
            (-1 means a pick not assigned to a local expert).
        dispatched_scores: Routing scores ``[num_recv_tokens, topk]``.

    Returns:
        permuted_hidden_states: ``[num_all_tokens, hidden]`` sorted by expert.
        permuted_scores: ``[num_all_tokens]`` scores in the same order.
        permuted_indices: ``[num_all_tokens]`` original token index for un-permute.
    """
    mask = dispatched_indices != -1
    valid_expert_ids = dispatched_indices[mask]  # 1d tensor
    valid_scores = dispatched_scores[mask]

    # Repeat each token by its valid count and select tokens in expert order.
    sort_order = torch.argsort(valid_expert_ids, stable=True)
    # size(0), not len(): under torch.compile the received-token count is an unbacked
    # symint (the compact dispatch layout is data-dependent), and len() forces it to a
    # Python int, which raises GuardOnDataDependentSymNode.
    permuted_indices = torch.arange(
        hidden_states.size(0), device=hidden_states.device
    ).repeat_interleave(mask.sum(dim=1))[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]

    return permuted_hidden_states, permuted_scores, permuted_indices


def _unpermute_tokens(
    permuted_hidden_states: torch.Tensor,
    permuted_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Reverse ``_permute_tokens``: scatter-add expert outputs back to unique tokens."""
    hidden_dim = permuted_hidden_states.shape[1]
    output_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=permuted_hidden_states.dtype,
        device=permuted_hidden_states.device,
    )
    output_hidden_states.scatter_add_(
        0, permuted_indices.unsqueeze(1).expand(-1, hidden_dim), permuted_hidden_states
    )
    return output_hidden_states


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor used to retrieve the cached EPHandle
    num_recv_tokens: int
    # Original local token count (combine's output length; not recoverable from the
    # permuted/deduplicated combine input, so it is carried explicitly for the fake impl).
    num_tokens: int
    cuda_graph_compatible: bool = False
    # Compact path (cuda_graph_compatible=False): gather/scatter mapping for the grouped GEMM.
    permuted_indices: torch.Tensor | None = None
    permuted_scores: torch.Tensor | None = None
    # Expand path (cuda_graph_compatible=True): per-received-row routing scores.
    recv_scores: torch.Tensor | None = None


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    num_local_experts: int,
    num_experts: int,
    *,
    num_tokens_per_rank: int,
    cuda_graph_compatible: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via DeepEP v2 ``ElasticBuffer``.

    Returns tokens in expert-major order for the grouped-GEMM expert path. In compact mode
    (``cuda_graph_compatible=False``) the deduplicated dispatch output is gathered by ``_permute_tokens``
    (matching the v1 path for identical numerics); in expand mode (``cuda_graph_compatible=True``) the
    static layout is already expert-grouped. Routing scores are applied to the expert outputs
    in ``combine_tokens``.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_dim]
        selected_experts_indices: Expert indices per token [num_tokens, top_k]
        top_scores: Routing scores per token [num_tokens, top_k]
        num_local_experts: Number of experts on this rank
        num_experts: Total number of experts across all ranks
        num_tokens_per_rank: Current number of local tokens. With uneven
            sharding, this must be the maximum local token count across EP
            ranks. DeepEP uses it as the per-rank layout stride and, in expand
            mode, to size the current ``recv_x``. This is distinct from the
            lifetime maximum used to initialize the communication buffer and
            must not exceed that maximum.
        cuda_graph_compatible: If True, use the static, no-host-sync expand layout so the forward is
            CUDA-graph-capturable (inference only -- both prefill and decode -- no backward);
            note it is forced False whenever grad is enabled. If False, use the compact
            layout with a host sync and full autograd (training).

    Returns:
        (routed_tokens [num_recv, hidden], tokens_per_expert [num_local_experts], state)
    """
    # The expand layout is inference-only ("must not be backward"), so gate it on a
    # no-grad context. With a single model_spec shared by trainer and generator, this
    # auto-selects: the trainer (autograd enabled) takes the compact path, while the
    # generator -- which runs the forward under torch.no_grad()/inference_mode -- takes
    # the CUDA-graph-compatible expand path. A cuda_graph_compatible=True spec used in a grad context
    # safely falls back to compact rather than hitting the no-backward kernel error.
    cuda_graph_compatible = cuda_graph_compatible and not torch.is_grad_enabled()

    buffer = _buffer
    assert buffer is not None, "Buffer must be initialized before dispatch"
    # Callers pass x.shape[0]. Pin it to a Python int before it crosses the op boundary:
    # the schema declares an int, and this value indexes a buffer preallocated to a fixed
    # num_max_tokens_per_rank, so a symbolic token count could not be served anyway. (The
    # unhashable-SymInt failure under compile came from the OUTPUT, not this argument --
    # see the dynamic_output_shape tag on the schema.)
    num_tokens_per_rank = int(num_tokens_per_rank)
    assert num_tokens_per_rank <= buffer.num_max_tokens_per_rank, (
        "DeepEP current token count "
        f"{num_tokens_per_rank} exceeds the "
        f"preallocated capacity of {buffer.num_max_tokens_per_rank}."
    )

    selected_experts_indices = selected_experts_indices.contiguous()
    top_scores = top_scores.contiguous()
    # Mask out zero-score selections (DeepEP uses -1 for "no selection").
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)
    if top_scores.dtype != torch.float32:
        top_scores = top_scores.float()

    (
        recv_x,
        recv_topk_idx,
        recv_scores,
        num_recv_per_expert,
        handle_id,
    ) = torch.ops.deepep.dispatch(
        hidden_states,
        selected_experts_indices,
        top_scores,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        num_tokens_per_rank=num_tokens_per_rank,
        cuda_graph_compatible=cuda_graph_compatible,
    )

    num_tokens_per_expert = num_recv_per_expert.to(recv_x.device)

    if cuda_graph_compatible:
        # Expand layout is already expert-grouped; feed it straight to the grouped GEMM.
        state = DispatchState(
            handle_id=handle_id,
            num_recv_tokens=recv_x.shape[0],
            num_tokens=num_tokens_per_rank,
            cuda_graph_compatible=True,
            recv_scores=recv_scores,
        )
        return recv_x, num_tokens_per_expert, state

    # Compact layout is deduplicated; gather into expert-major order (plain autograd, so the
    # gather's gradient flows back through the dispatch custom op's combine-backward).
    num_recv_tokens = recv_x.shape[0]
    routed_input, permuted_scores, permuted_indices = _permute_tokens(
        recv_x, recv_topk_idx, recv_scores
    )
    state = DispatchState(
        handle_id=handle_id,
        num_recv_tokens=num_recv_tokens,
        num_tokens=num_tokens_per_rank,
        cuda_graph_compatible=False,
        permuted_indices=permuted_indices,
        permuted_scores=permuted_scores,
    )
    return routed_input, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
) -> torch.Tensor:
    """Combine expert outputs back to tokens via DeepEP v2.

    Routing scores are applied here (in plain PyTorch) before the pure-reduction combine
    op, so autograd handles the score gradient. Combine is async; the caller MUST call
    ``sync_combine()`` before using the result.

    Compact (``cuda_graph_compatible=False``): weight each expert-major row, scatter-add back to the
    deduplicated tokens (``_unpermute_tokens``), then combine -- matching the v1 path.
    Expand (``cuda_graph_compatible=True``): weight per received row, then combine over the static
    layout (the handle drives the expand reduction).

    Args:
        hidden_states: Raw (unweighted) expert outputs [num_recv, hidden].
        state: Dispatch state from ``dispatch_tokens``.

    Returns:
        Combined tokens [num_tokens, hidden_dim].
    """
    # Outer grad state decides whether the combine op frees the handle itself (no
    # backward) or leaves it for combine-backward. Evaluated here, before the op.
    will_backward = torch.is_grad_enabled()

    if not state.cuda_graph_compatible:
        assert state.permuted_indices is not None
        if state.permuted_scores is not None:
            hidden_states = hidden_states * state.permuted_scores.to(
                hidden_states.dtype
            ).reshape(-1, 1)
        hidden_states = _unpermute_tokens(
            hidden_states, state.permuted_indices, state.num_recv_tokens
        )
        return torch.ops.deepep.combine(
            hidden_states, state.handle_id, state.num_tokens, will_backward
        )

    if state.recv_scores is not None:
        # One routing score per received row (each row is one token->expert assignment).
        # Collapse the trailing dim with sum so this is correct whether recv_scores is
        # [num_recv], [num_recv, 1], or [num_recv, topk] with a single valid entry/row.
        per_row_score = state.recv_scores.reshape(hidden_states.shape[0], -1).sum(
            dim=-1, keepdim=True
        )
        hidden_states = hidden_states * per_row_score.to(hidden_states.dtype)
    return torch.ops.deepep.combine(
        hidden_states, state.handle_id, state.num_tokens, will_backward
    )
