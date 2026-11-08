# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The generator side of RL routing replay (R3, https://arxiv.org/abs/2510.11370).

vLLM records each token's experts per layer when ``enable_return_routed_experts``
is on: it binds a ``capture_fn`` to every module with a ``layer_id`` and a
``capture_fn`` attribute, and returns the ids per finished request
(``CompletionOutput.routed_experts`` ``[prompt + generated - 1, num_layers, top_k]``).
Here each MoE router becomes such a module: a forward hook passes its expert ids
through ``torchtitan_rl::capture_routed_experts``, opaque to dynamo and eager
under CUDA graph capture, which calls the bound ``capture_fn``.
"""

import torch
from torch import nn

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter

# layer index -> router, for the op below
_CAPTURE_SOURCES: dict[int, nn.Module] = {}


def records_routed_experts(model_config: Decoder.Config) -> bool:
    """Whether the model's routers replay or measure the generator's experts."""
    return any(
        router.routing_replay != "off"
        for _fqn, router, _parent, _attr in model_config.traverse(
            TokenChoiceTopKRouter.Config
        )
    )


@torch.library.custom_op("torchtitan_rl::capture_routed_experts", mutates_args=())
def _capture_routed_experts(
    topk_expert_ids_TK: torch.Tensor, layer_idx: torch.Tensor
) -> torch.Tensor:
    """Hand layer ``layer_idx``'s (a CPU scalar) expert ids to vLLM's capturer;
    returns a copy that the MoE reads, so a compiled graph keeps this op."""
    capture_fn = _CAPTURE_SOURCES[int(layer_idx)].capture_fn
    if capture_fn is not None:
        capture_fn(topk_expert_ids_TK)
    return topk_expert_ids_TK.clone()


@_capture_routed_experts.register_fake
def _(topk_expert_ids_TK, layer_idx) -> torch.Tensor:
    return torch.empty_like(topk_expert_ids_TK)


def _capture_hook(router: nn.Module, args, output):
    topk_scores_TK, topk_expert_ids_TK, routing_map_TE = output
    return (
        topk_scores_TK,
        torch.ops.torchtitan_rl.capture_routed_experts(
            topk_expert_ids_TK, router.layer_idx_cpu
        ),
        routing_map_TE,
    )


def install_routed_experts_capture(model: Decoder) -> None:
    """Make each MoE block's router a vLLM capture source, keyed by its block index
    (= the decoder layer vLLM's buffer rows follow; dense blocks keep zeros)."""
    _CAPTURE_SOURCES.clear()
    for layer_id, layer in model.layers.items():
        moe = getattr(layer, "feed_forward", None)
        if not isinstance(moe, MoE):
            continue
        router = moe.router
        router.layer_id = int(layer_id)
        router.capture_fn = None
        # explicit CPU: vLLM builds the model under a CUDA default device
        router.layer_idx_cpu = torch.tensor(
            int(layer_id), dtype=torch.int64, device="cpu"
        )
        router.register_forward_hook(_capture_hook)
        _CAPTURE_SOURCES[int(layer_id)] = router
    if not _CAPTURE_SOURCES:
        raise ValueError("routing replay: the generator model has no MoE block")
