# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.models.inputs import MoEInputsDict, MTPInputsDict, TransformerInputsDict
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(
    pred: torch.Tensor | list[torch.Tensor] | TransformerInputsDict,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    if isinstance(pred, dict):
        pred = pred["tokens_list"]
    if isinstance(pred, list):
        pred = pred[0]
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def multi_token_cross_entropy_loss(
    preds: list[torch.Tensor] | MTPInputsDict,
    labels: torch.Tensor,
    loss_fn: LossFunction,
    job_config: JobConfig,
) -> torch.Tensor:
    """Multi-token cross-entropy loss function for Transformer model training.

    Based on DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437.
    """
    if isinstance(preds, dict):
        preds = preds["tokens_list"]
    assert isinstance(preds, list)
    main_loss = loss_fn(preds[0], labels[:, : job_config.training.seq_len])

    mtp_loss = 0
    for label_offset, pred in enumerate(preds[1:], 1):
        loss = loss_fn(
            pred,
            labels[:, label_offset : label_offset + job_config.training.seq_len],
        )
        # Take average over MTP predictions.
        loss = loss / job_config.training.num_mtp_tokens
        mtp_loss = mtp_loss + loss
    return main_loss + mtp_loss * job_config.training.mtp_loss_weight


def moe_loss(
    pred: MoEInputsDict,
    labels: torch.Tensor,
    loss_fn: LossFunction,
) -> torch.Tensor:
    """Sequence-wise auxiliary loss-enhanced loss function for MoE Transformer
    model training.
    """
    loss = loss_fn(pred, labels)
    if isinstance(pred, dict):
        loss += pred["aux_loss"]
    return loss


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn
