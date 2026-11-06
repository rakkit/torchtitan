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


def test_mtp_loss_runs_chunked_and_reports_its_terms():
    """The run's loss: ChunkedLossWrapper(MTPCrossEntropyLoss) over several
    chunks equals the loss on the whole logits, and its metrics are the two
    terms (summed over the chunks, as their suffix says)."""
    from torch import nn

    from torchtitan.components.loss import ChunkedLossWrapper

    torch.manual_seed(0)
    t, d = 64, 16
    head = nn.Linear(d, VOCAB, bias=False)
    hidden = (torch.randn(t, d), torch.randn(t, d))
    labels = tuple(torch.randint(0, VOCAB, (t,)) for _ in range(2))
    labels[1][-1] = IGNORE_INDEX
    counts = torch.stack([(lab != IGNORE_INDEX).sum() for lab in labels])
    mtp = MTPCrossEntropyLoss(MTPCrossEntropyLoss.Config(mtp_scale=0.3))
    chunked = ChunkedLossWrapper(
        ChunkedLossWrapper.Config(
            num_chunks=4, loss_fn=MTPCrossEntropyLoss.Config(mtp_scale=0.3)
        )
    )
    chunked.lm_head = head
    loss, metrics = chunked(hidden, labels, counts)
    want, want_metrics = mtp(tuple(head(h) for h in hidden), labels, counts)
    torch.testing.assert_close(loss, want)
    assert (
        metrics.keys()
        == want_metrics.keys()
        == {
            "mtp/main_loss_mean",
            "mtp/depth_1_loss_mean",
        }
    )
    for key in metrics:
        torch.testing.assert_close(metrics[key], want_metrics[key])


def test_the_mtp_block_mirrors_the_last_layer_with_its_own_expert_seed():
    with torch.device("meta"):
        cfg = build_model_config(
            "mis-8b-proxy",
            parallelism=ParallelismConfig(),
            seq_len=128,
            n_layers=2,
            swa_pattern="SF",
            rope_pattern=None,
            num_mtp_layers=1,
        )
    last, block = cfg.layers[-1], cfg.mtp_layers[0]
    assert type(block.attention.inner_attention) is type(last.attention.inner_attention)
    model = cfg.build()  # CPU: the expert inits are seeded per layer id
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    a = model.layers["1"].feed_forward.routed_experts.w13.weight
    b = model.mtp_layers[0].feed_forward.routed_experts.w13.weight
    assert a.shape == b.shape and not torch.equal(a, b)
    with pytest.raises(ValueError, match="full attention"):
        build_model_config(
            "mis-8b-proxy",
            parallelism=ParallelismConfig(),
            n_layers=2,
            swa_pattern="FS",
            rope_pattern=None,
            num_mtp_layers=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex / conv need CUDA")
def test_eval_mode_skips_the_mtp_depth():
    positions = torch.cat([torch.arange(n) for n in DOCS]).cuda()
    tokens = torch.randint(0, VOCAB, (positions.numel(),), device="cuda")
    model = _model(1)
    with torch.no_grad():
        train_out = _forward(model, tokens, positions)
        model.eval()
        eval_out = _forward(model, tokens, positions)
    assert len(train_out) == 2 and len(eval_out) == 1
    assert torch.equal(eval_out[0], train_out[0])


def test_parallelize_treats_the_mtp_block_as_a_decoder_block(tmp_path):
    """FSDP wraps the MTP block and activation remat checkpoints it, as a
    decoder block (both walk `layers`); `layers` holds the decoder blocks
    only afterwards. With compile_blocks every block, the MTP one included,
    is torch.compile'd and its linears drop the in-place WGRAD accumulation a
    traced backward cannot do; the LM head, outside the blocks, keeps it.
    Fake process group, CPU: no collective runs, nothing is traced. (One
    parallelize per process: a second one in the same process fails on a mesh
    left from the first.)"""
    from unittest import mock

    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    from torchtitan.config import PrecisionConfig
    from torchtitan.distributed import ParallelismContext
    from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
    from torchtitan.distributed.local_compile import LocalCompileConfig

    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        parallelism = ParallelismConfig(data_parallel_shard_degree=2)
        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        with mock.patch(
            "torchtitan.distributed.parallelism_context.device_type", "cpu"
        ):
            pc.build_mesh()
        cfg = build_model_config(
            "mis-8b-proxy",
            parallelism=parallelism,
            seq_len=128,
            n_layers=2,
            swa_pattern="SF",
            rope_pattern=None,
            num_mtp_layers=1,
        )
        cfg.compile_blocks = True
        with torch.device("meta"):
            model = cfg.build()
        block = model.mtp_layers[0]
        plain_forward = block.forward
        model.parallelize(
            parallelism_context=pc,
            precision=PrecisionConfig(),
            parallelism=parallelism,
            local_compile=LocalCompileConfig(regions=["loss"]),
            activation_remat=DefaultActivationRematConfig(),
            dump_folder=str(tmp_path),
        )
        assert list(model.layers.keys()) == ["0", "1"]
        assert "FSDP" in type(block).__name__
        assert block.forward is not plain_forward  # remat.checkpoint's wrapper
        for b in [*model.layers.values(), block]:
            assert b._compiled_call_impl is not None
            flags = [
                m.inplace_wgrad_accum
                for m in b.modules()
                if hasattr(m, "inplace_wgrad_accum")
            ]
            assert flags and not any(flags)
        assert model.lm_head.inplace_wgrad_accum
    finally:
        dist.destroy_process_group()
