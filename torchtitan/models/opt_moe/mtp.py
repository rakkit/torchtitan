# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-token prediction for OPT MoE (DeepSeek-V3's MTP objective; opt_moe
imports no deepseek_v3 code): the token roll that builds each MTP depth's
inputs and labels, and the weighted cross-entropy over the main and MTP
predictions. The MTP block and the model's MTP forward live in ``model.py``.

Tensor suffixes: ``T`` tokens."""

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch

from torchtitan.components.loss import CrossEntropyLoss, Loss


def roll_mtp_sequence(
    sequence_T: torch.Tensor,
    *,
    shift: int,
    fill_value: int | bool,
    positions_T: torch.Tensor | None = None,
    padding_mask_T: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sequence_T`` rolled left by ``shift`` tokens, without crossing a
    document (a position that does not continue by ``shift``) or padding, and
    the mask of the positions that received a token (the rest hold
    ``fill_value``). The same roll as deepseek_v3's ``roll_mtp_sequence``."""
    num_tokens = sequence_T.shape[0]
    if shift <= 0 or shift > num_tokens:
        raise ValueError(f"MTP roll shift must be in [1, {num_tokens}], got {shift}.")
    shifted_T = torch.full_like(sequence_T, fill_value)
    valid_T = torch.zeros_like(sequence_T, dtype=torch.bool)
    keep = torch.ones_like(sequence_T[shift:], dtype=torch.bool)
    if positions_T is not None:
        keep &= positions_T[shift:] == positions_T[: num_tokens - shift] + shift
    if padding_mask_T is not None:
        keep &= ~padding_mask_T[: num_tokens - shift] & ~padding_mask_T[shift:]
    with spmd.no_typecheck():
        valid_T[: num_tokens - shift] = keep
    shifted_T[: num_tokens - shift] = torch.where(
        keep, sequence_T[shift:], shifted_T[: num_tokens - shift]
    )
    return shifted_T, valid_T


class MTPCrossEntropyLoss(Loss[torch.Tensor | tuple[torch.Tensor, ...]]):
    """``CE(main) + mtp_scale * mean over depths CE(depth)``: the main
    prediction and one per MTP depth, each against its own labels and its own
    global token count (the collator counts each depth's targets). Metrics:
    ``mtp/main_loss`` and ``mtp/depth_<d>_loss`` (normalized as the loss, so the
    trainer sums them over microbatches and ranks). On one prediction (the
    validator's main objective) it is the cross-entropy."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        mtp_scale: float = 0.3
        """The MTP terms' total weight (DeepSeek-V3's lambda: 0.3, later 0.1)."""
        global_vocab_size: int | None = None

    def __init__(self, config: Config):
        self.cross_entropy_loss = CrossEntropyLoss(
            CrossEntropyLoss.Config(global_vocab_size=config.global_vocab_size)
        )
        self.mtp_scale = config.mtp_scale

    def __call__(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del loss_inputs
        if isinstance(pred, torch.Tensor) and isinstance(labels, torch.Tensor):
            return self.cross_entropy_loss(pred, labels, global_loss_token_counts)
        if not isinstance(pred, tuple) or not isinstance(labels, tuple):
            raise ValueError(
                "MTPCrossEntropyLoss expects a prediction / labels tensor pair or tuples"
            )
        if len(pred) != len(labels) or len(pred) < 2:
            raise ValueError(
                "MTPCrossEntropyLoss needs the main and >= 1 MTP prediction, one "
                f"labels tensor each: got {len(pred)} predictions, {len(labels)} labels"
            )
        if global_loss_token_counts is not None and (
            global_loss_token_counts.ndim != 1
            or global_loss_token_counts.numel() != len(pred)
        ):
            raise ValueError(
                "MTPCrossEntropyLoss needs one global token count per prediction"
            )
        if global_loss_token_counts is None:
            counts: tuple[torch.Tensor | None, ...] = (None,) * len(pred)
        else:
            with spmd.no_typecheck():
                counts = tuple(
                    c.clamp_min(1) for c in global_loss_token_counts.unbind()
                )
        main_loss, _ = self.cross_entropy_loss(pred[0], labels[0], counts[0])
        metrics = {"mtp/main_loss": main_loss.detach()}
        mtp_weight = self.mtp_scale / (len(pred) - 1)
        mtp_loss = pred[0].new_zeros((), dtype=torch.float32)
        if spmd.is_type_checking():
            mtp_loss = spmd.mutate_type(
                mtp_loss, src=spmd.R, dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I}
            )
        for depth in range(1, len(pred)):
            depth_loss, _ = self.cross_entropy_loss(
                pred[depth], labels[depth], counts[depth]
            )
            metrics[f"mtp/depth_{depth}_loss"] = depth_loss.detach()
            mtp_loss = mtp_loss + depth_loss * mtp_weight
        return main_loss + mtp_loss, metrics
