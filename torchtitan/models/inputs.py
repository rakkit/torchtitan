# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NotRequired, TypedDict

import torch


class TransformerInputsDict(TypedDict):
    tokens_list: list[torch.Tensor | None] | torch.Tensor
    start_pos: NotRequired[int]


TransformerInputs = torch.Tensor | TransformerInputsDict


class MTPInputsDict(TransformerInputsDict):
    prev_embed: NotRequired[torch.Tensor | None]


MTPInputs = torch.Tensor | MTPInputsDict


class MoEInputsDict(TransformerInputsDict):
    tokens_list: list[torch.Tensor | None] | torch.Tensor
    aux_loss: NotRequired[torch.Tensor]


MoEInputs = torch.Tensor | MoEInputsDict
