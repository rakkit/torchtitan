# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The LM-head geometry takes the head's input from the training forward:
under the chunked loss the model's output is that input, so the engine keeps
a few rows of it (no second forward), and DiSCO uses them for one step."""

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO
from torchtitan.training_engine import sample_rows, TrainingEngine


def _engine(pred, *, output_is_head_input=True):
    engine = object.__new__(TrainingEngine)
    engine._output_is_head_input = output_is_head_input
    engine._head_input_rows, engine._head_input_seed = 0, 0
    engine._head_input = None
    engine.config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=True),
        debug=SimpleNamespace(spmd_typechecking=False),
    )
    engine.parallel_dims = None
    w = nn.Parameter(torch.ones(()))
    engine.model_parts = [lambda inputs, **kw: pred * w]
    engine.loss_fn = lambda out, labels, **kw: (out.sum(), {})
    return engine


def _forward_backward(engine):
    engine._non_pp_forward_backward_body(
        inputs=None, labels=None, model_kwargs={}, loss_kwargs={}
    )


def test_the_engine_keeps_rows_of_the_forward_output():
    pred = torch.randn(64, 8)
    engine = _engine(pred)
    engine.keep_head_input_rows(16, seed=3)
    _forward_backward(engine)
    rows = engine.take_head_input_rows()
    assert rows.shape == (16, 8) and rows.dtype == torch.float32
    assert torch.equal(rows, sample_rows(pred, 16, 3))
    assert all((pred == r).all(dim=1).any() for r in rows)
    assert engine.take_head_input_rows() is None  # one step only


def test_nothing_is_kept_unless_asked_or_without_the_chunked_loss():
    engine = _engine(torch.randn(64, 8))
    _forward_backward(engine)
    assert engine.take_head_input_rows() is None
    engine = _engine(torch.randn(64, 8), output_is_head_input=False)
    engine.keep_head_input_rows(16, seed=3)
    _forward_backward(engine)
    assert engine.take_head_input_rows() is None


def test_the_rows_depend_on_the_seed_only():
    x = torch.randn(100, 4)
    assert torch.equal(sample_rows(x, 10, 5), sample_rows(x, 10, 5))
    assert not torch.equal(sample_rows(x, 10, 5), sample_rows(x, 10, 6))


@pytest.fixture
def one_rank():
    if dist.is_initialized():  # another module's one-rank group
        yield
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


class _NoParallel:
    dp_replicate_enabled = dp_shard_enabled = cp_enabled = dp_cp_enabled = False
    fsdp_enabled = tp_enabled = pp_enabled = ep_enabled = etp_enabled = False
    world_mesh = None

    def get_optional_mesh(self, dims):
        return None

    def get_mesh(self, dims):
        return None


def _one_matrix_disco():
    w = nn.Parameter(torch.randn(8, 4))
    opt = DiSCO(
        [{"params": [w], "param_names": ["w"]}],
        parallel_dims=_NoParallel(),
        lr=0.1,
        args={"steps": 3},
        param_layouts={"w": ParamLayout("matrix")},
    )
    return opt, w


def test_disco_step_consumes_the_norm_arm(one_rank):
    """One arm covers one step: the real `step()` must clear it."""
    opt, w = _one_matrix_disco()
    opt.calculate_norm_at_next_step()
    w.grad = torch.randn_like(w)
    opt.step()
    assert not opt.need_to_calculate_norm


def test_disco_drops_the_head_input_after_its_step(one_rank):
    """Single use: a later norm step must bring its own."""
    opt, w = _one_matrix_disco()
    opt.head_probe_h = torch.randn(2, 4)
    w.grad = torch.randn_like(w)
    opt.step()
    assert opt.head_probe_h is None
