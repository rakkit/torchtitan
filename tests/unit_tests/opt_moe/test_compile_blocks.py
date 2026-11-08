# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""opt_moe's compile_blocks through a run's pattern: training steps of two
accumulated microbatches, a validation pass in between (eval mode, no_grad, its
own token count), training again; the k / v conv, the gate, an MTP block, the
recipe's w2 remat. One GPU, fp32.

- Every block type of the production shapes (a dense full-attention layer 0,
  sliding-window MoE layers, a full MoE layer without RoPE, the MTP block) stays
  inside dynamo's recompile budget: a 6-layer run failed with "Dynamo recompile
  limit exceeded" at its first training step after validation. Each block type
  compiles a training graph, a no_grad validation graph and a dynamic-shape
  training graph (a microbatch with other document and token counts, as
  NodeComputeBalancer's rows): 11 entries here, over torch's default limit of 8
  (this test fails with it); a third token count reuses the dynamic graph.
- The compiled blocks match eager blocks in the same setup (FSDP, the w2
  remat; its eager path through the MTP block read a remat placeholder): the
  loss and every gradient of the accumulated step, and the validation logits. Dense feed-forwards here: an MoE
  router's top-k flips on a near-tie between the two roundings (measured: one
  token of 2,048), which moves that layer's gradients by percents."""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.testing._internal.common_distributed import TIMEOUT_OVERRIDE
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.opt_moe import build_model_config
from torchtitan.models.opt_moe.mtp import roll_mtp_sequence

T, T_VAL, V = 512, 384, 512
# the accumulated microbatches pack different document and token counts (real
# data; NodeComputeBalancer's variable rows): the second one recompiles every
# block type with dynamic shapes, the third reuses that graph
DOCS, DOCS_2, DOCS_3, DOCS_VAL = [200, 1, 311], [100, 348], [60, 500], [384]
# The largest |compiled - eager| / |eager| (fp32: the compiled blocks fuse and
# reorder the same ops).
TOL = 1e-4
TIMEOUT_OVERRIDE["test_training_with_validation_matches_eager"] = 1500
TIMEOUT_OVERRIDE["test_training_with_validation_stays_in_the_recompile_budget"] = 1500


def _config(*, moe: bool, **overrides):
    return build_model_config(
        "mis-8b-proxy",
        parallelism=ParallelismConfig(),
        seq_len=T,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        swa_pattern="FSSSSF",
        rope_pattern="RRRRRN",
        n_dense_layers=1 if moe else 6,
        dense_hidden_dim=256,
        num_experts=8,
        top_k=2,
        moe_hidden_dim=64,
        conv_kernel_size=4,
        gated_attention_type="element-wise",
        num_mtp_layers=1,
        vocab_size=V,
        # the production recipe's router init: scores far apart, so the eager and
        # compiled rounding cannot flip a token's top-k
        router_init_fn_type="scaled_orthogonal",
        router_init_std=4.0,
        **overrides,
    )


def _batch(docs, seed):
    g = torch.Generator().manual_seed(seed)
    n = sum(docs)
    positions = torch.cat([torch.arange(d) for d in docs]).cuda()
    tokens = torch.randint(0, V, (n,), generator=g).cuda()
    labels = torch.randint(0, V, (n,), generator=g).cuda()
    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=0, positions_T=positions
    )
    return tokens, labels, rolled, valid, positions


def _loss(model, batch):
    tokens, labels, rolled, valid, positions = batch
    predictions = model(
        (tokens, rolled),
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
        mtp_input_valid_masks=(valid,),
    )
    return sum(F.cross_entropy(p.float(), labels) for p in predictions)


def _validation_logits(model, batch):
    tokens, _, _, _, positions = batch
    model.eval()
    with torch.no_grad():
        logits = model(
            (tokens, tokens),
            positions=positions,
            attention_metadata=model._get_attention_metadata(positions),
            mtp_input_valid_masks=(torch.ones_like(tokens, dtype=torch.bool),),
        )
    model.train()
    return logits[0]  # eval mode: the main prediction only


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestCompileBlocks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    def _models(self, *, moe: bool, **overrides):
        torch.manual_seed(0)
        with torch.device("meta"):
            reference = _config(moe=moe, **overrides).build()
        reference.to_empty(device="cuda")
        with torch.no_grad():
            reference.init_states(buffer_device=torch.device("cuda"))

        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        pc.build_mesh()
        model = self._parallelized(
            pc, reference, moe=moe, compile_blocks=True, **overrides
        )
        return pc, reference, model

    def _parallelized(
        self, pc, reference, *, moe: bool, compile_blocks: bool, **overrides
    ):
        """``reference``'s weights in the recipe's setup: FSDP, the w2 remat,
        compile_blocks on or off."""
        cfg = _config(moe=moe, **overrides)
        cfg.compile_blocks = compile_blocks
        with torch.device("meta"):
            model = cfg.build()
        model.parallelize(
            parallelism_context=pc,
            precision=PrecisionConfig(param_dtype="float32", reduce_dtype="float32"),
            parallelism=ParallelismConfig(),
            local_compile=LocalCompileConfig(regions=[]),
            activation_remat=ActivationRematConfig(
                save_regions=["*"],
                recompute_regions=["*routed_experts.w13.*", "*routed_experts.w2.*"],
            ),
            dump_folder="",
        )
        model.to_empty(device="cuda")
        with torch.no_grad(), pc.activate_spmd():
            model.init_states(buffer_device=torch.device("cuda"))
        set_model_state_dict(
            model,
            reference.state_dict(),
            options=StateDictOptions(full_state_dict=True),
        )
        return model

    @with_comms
    def test_training_with_validation_stays_in_the_recompile_budget(self):
        pc, _, model = self._models(moe=True)
        train = [_batch(DOCS, 1), _batch(DOCS_2, 2), _batch(DOCS_3, 4)]
        val = _batch(DOCS_VAL, 3)
        with pc.activate_spmd():
            for _ in range(3):
                for batch in train:
                    _loss(model, batch).backward()
                _validation_logits(model, val)

    @with_comms
    def test_varlen_document_lengths_do_not_recompile(self):
        # The varlen backend takes the longest document as an int. Under CP a
        # rank's value can stay fixed for many steps (clamped to its chunk);
        # its first new value recompiled that rank's blocks while every other
        # rank waited in a collective (12 s steps at 256k).
        pc, _, model = self._models(moe=False, attn_backend="varlen")
        with pc.activate_spmd():
            for _ in range(2):
                _loss(model, _batch([256, 256], 1)).backward()
            with torch._dynamo.config.patch(error_on_recompile=True):
                for docs in ([100, 412], [12, 500]):
                    _loss(model, _batch(docs, 2)).backward()

    @with_comms
    def test_training_with_validation_matches_eager(self):
        pc, reference, model = self._models(moe=False)
        # the reference: the same setup with eager blocks (the w2 remat's eager
        # path through the MTP block included)
        eager = self._parallelized(pc, reference, moe=False, compile_blocks=False)
        # Cleared as the trainer's optimizer does, on the parameters held from
        # the start: after a no_grad forward FSDP leaves the [norm, lm_head]
        # group unsharded, and model.zero_grad() would clear those tensors
        # instead of the sharded gradients.
        params = [*eager.parameters(), *model.parameters()]
        train = [_batch(DOCS, 1), _batch(DOCS_2, 2)]
        val = _batch(DOCS_VAL, 3)
        for step in range(3):
            for p in params:
                p.grad = None
            with pc.activate_spmd():
                for batch in train:  # two accumulated microbatches
                    want = _loss(eager, batch)
                    want.backward()
                    got = _loss(model, batch)
                    got.backward()
                    rel = ((got - want).abs() / want.abs()).item()
                    self.assertLess(rel, TOL, msg=f"step {step} loss")
            grads = {
                m: {n: p.grad.full_tensor() for n, p in m.named_parameters()}
                for m in (eager, model)
            }
            errors = {
                n: ((grads[model][n] - g).norm() / g.norm()).item()
                for n, g in grads[eager].items()
            }
            worst = sorted(errors.items(), key=lambda kv: -kv[1])[:6]
            print(f"step {step} worst gradient errors: {worst}", flush=True)
            self.assertLess(worst[0][1], TOL, msg=f"step {step} {worst}")
            # validation between the training steps, as a run's validator
            with pc.activate_spmd():
                want_val = _validation_logits(eager, val)
                got_val = _validation_logits(model, val)
            rel = ((got_val - want_val).norm() / want_val.norm()).item()
            self.assertLess(rel, TOL, msg=f"step {step} validation")
