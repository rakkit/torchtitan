# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe multi-token prediction (DeepSeek-V3's MTP, one depth): the token roll
stays inside documents, MTP leaves the main prediction unchanged, the depth's
prediction at token t reads only tokens <= t + 1 of t's document, and the MTP
loss trains the MTP block."""

import pytest
import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.mtp import MTPCrossEntropyLoss, roll_mtp_sequence

DOCS = [37, 1, 90]
VOCAB = 512


def test_roll_stays_inside_documents_and_off_padding():
    positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    padding = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
    tokens = torch.arange(10, 19)
    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=-1, positions_T=positions, padding_mask_T=padding
    )
    # token t gets t + 1 only where t + 1 continues t's document and neither is padding
    assert rolled.tolist() == [11, 12, -1, 14, -1, 16, -1, -1, -1]
    assert valid.tolist() == [True, True, False, True, False, True, False, False, False]


def _model(num_mtp_layers: int):
    cfg = build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=128,
        n_layers=2,
        swa_pattern="SF",
        rope_pattern=None,
        vocab_size=VOCAB,
        conv_kernel_size=4,
        num_mtp_layers=num_mtp_layers,
    )
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    return model


def _forward(model, tokens, positions):
    metadata = model._get_attention_metadata(positions)
    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=0, positions_T=positions
    )
    return model(
        (tokens, rolled),
        positions=positions,
        attention_metadata=metadata,
        mtp_input_valid_masks=(valid,),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex / conv need CUDA")
def test_mtp_keeps_the_main_prediction_and_trains_its_block():
    positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (positions.numel(),), device="cuda")
    labels = torch.randint(0, VOCAB, (positions.numel(),), device="cuda")

    plain = _model(0)
    mtp = _model(1)
    missing, unexpected = mtp.load_state_dict(plain.state_dict(), strict=False)
    assert (
        not unexpected and missing and all(k.startswith("mtp_layers.") for k in missing)
    )
    with torch.no_grad():
        want = plain(
            tokens,
            positions=positions,
            attention_metadata=plain._get_attention_metadata(positions),
        )
        main, depth1 = _forward(mtp, tokens, positions)
    assert torch.equal(main, want)  # MTP adds a prediction, changes none

    # depth 1 at token t reads tokens <= t + 1 of t's document: change token 20
    # (document 0) -> rows 0..18 unchanged, and no row of the later documents
    changed = tokens.clone()
    changed[20] = (changed[20] + 1) % VOCAB
    with torch.no_grad():
        _, depth1_changed = _forward(mtp, changed, positions)
    diff = (depth1_changed - depth1).abs().amax(-1) > 0
    assert not diff[:19].any() and diff[19:21].all() and not diff[DOCS[0] :].any()

    mtp_labels, _ = roll_mtp_sequence(
        labels, shift=1, fill_value=IGNORE_INDEX, positions_T=positions
    )
    counts = torch.stack(
        [(labels != IGNORE_INDEX).sum(), (mtp_labels != IGNORE_INDEX).sum()]
    )
    loss, _ = MTPCrossEntropyLoss(MTPCrossEntropyLoss.Config(mtp_scale=0.3))(
        _forward(mtp, tokens, positions), (labels, mtp_labels), counts
    )
    loss.backward()
    for name, p in mtp.named_parameters():
        if name.startswith("mtp_layers.") and "expert_bias" not in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
