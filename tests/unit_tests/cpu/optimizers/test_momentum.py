# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A group's `momentum` beta is the memory weight (the averaged form), written
from its definition:

    buf_t = beta buf_(t-1) + (1 - beta) g_t,  buf_0 = 0
    step direction: buf_t, or with `nesterov` beta buf_t + (1 - beta) g_t

beta = 0 is the raw gradient; beta = 1 would never update and is refused, as is
any beta outside [0, 1). Observed through `sgd` with the identity compute, lr 1
and no decay: W_(t-1) - W_t is the direction."""

import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from torchtitan.models.common.param_layout import ParamLayout
from torchtitan.optimizers.disco.optimizer import DiSCO

M, N, STEPS = 6, 5, 4
NAME = "layers.0.feed_forward.w2.weight"


@pytest.fixture(scope="module", autouse=True)
def one_rank():
    if dist.is_initialized():
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


def _sgd(w, beta, nesterov):
    return DiSCO(
        [
            {
                "params": [w],
                "param_names": [NAME],
                "momentum": beta,
                "nesterov": nesterov,
            }
        ],
        parallelism_context=_NoParallel(),
        lr=1.0,
        rule="sgd",
        args={"pre": "identity", "compute": "identity", "scale": "one"},
        param_layouts={NAME: ParamLayout("matrix")},
    )


def _grads(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(M, N, generator=g, dtype=torch.float64) for _ in range(STEPS)]


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.5, 0.9, 0.95])
def test_the_direction_is_the_average_of_the_gradients(beta, nesterov):
    w = nn.Parameter(torch.randn(M, N, generator=torch.Generator().manual_seed(1)))
    opt = _sgd(w, beta, nesterov)
    buf = torch.zeros(M, N, dtype=torch.float64)
    for t, g in enumerate(_grads(), start=1):
        before = w.detach().clone()
        w.grad = g.float()
        opt.step()
        buf = beta * buf + (1 - beta) * g
        want = beta * buf + (1 - beta) * g if nesterov else buf
        got = (before - w.detach()).double()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=f"step {t}")
        if 0.0 < beta:
            torch.testing.assert_close(
                opt.state[w]["momentum_buffer"].double(), buf, rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("beta", [1.0, 1.5, -0.1])
def test_a_memory_weight_outside_0_1_is_refused(beta):
    w = nn.Parameter(torch.zeros(M, N))
    with pytest.raises(ValueError, match="momentum"):
        _sgd(w, beta, False)


def test_no_buffer_where_nothing_averages():
    """beta = 0 keeps no momentum buffer."""
    w = nn.Parameter(torch.randn(M, N, generator=torch.Generator().manual_seed(2)))
    opt = _sgd(w, 0.0, False)
    w.grad = torch.ones(M, N)
    opt.step()
    assert "momentum_buffer" not in opt.state[w]


class _Flat:
    """The checkpointer's view of one optimizer: the flat FQN-keyed state."""

    def __init__(self, opt):
        self.opt = opt

    def state_dict(self):
        from torchtitan.components.optim.utils import (
            get_flat_optim_state_dict,
            init_optim_state,
        )

        init_optim_state(self.opt)
        return get_flat_optim_state_dict(self.opt)

    def load_state_dict(self, sd):
        from torchtitan.components.optim.utils import (
            init_optim_state,
            load_flat_optim_state_dict,
        )

        init_optim_state(self.opt)
        for g in self.opt.param_groups:  # what the container supplies on load
            for k in DiSCO._CONFIG_ONLY_GROUP_KEYS:
                if k in g:
                    sd.setdefault(f"param_groups.{NAME}.{k}", g[k])
        load_flat_optim_state_dict(self.opt, sd)


def _resume(beta_saved, beta_live, path):
    import torch.distributed.checkpoint as dcp

    gen = torch.Generator().manual_seed(3)
    w = nn.Parameter(torch.randn(M, N, generator=gen))
    opt = _sgd(w, beta_saved, False)
    w.grad = torch.ones(M, N)
    opt.step()
    dcp.save({"optimizer": _Flat(opt)}, checkpoint_id=str(path))
    w2 = nn.Parameter(w.detach().clone())
    opt2 = _sgd(w2, beta_live, False)
    dcp.load({"optimizer": _Flat(opt2)}, checkpoint_id=str(path))
    return opt, opt2, w, w2


def test_a_resume_may_change_beta_while_the_group_averages(tmp_path):
    """Through DCP (the checkpointer's path): the buffer comes back, the
    config's beta runs from the next step on."""
    opt, opt2, w, w2 = _resume(0.9, 0.5, tmp_path)
    assert opt2.param_groups[0]["momentum"] == 0.5
    buf = opt.state[w]["momentum_buffer"]
    assert torch.equal(opt2.state[w2]["momentum_buffer"], buf)
    w2.grad = torch.ones(M, N)
    opt2.step()
    torch.testing.assert_close(
        opt2.state[w2]["momentum_buffer"], 0.5 * buf + 0.5 * torch.ones(M, N)
    )


def test_a_resume_cannot_turn_averaging_on(tmp_path):
    """A group that kept no buffer (beta 0, or amphione) resumed with beta in
    (0, 1): the checkpoint has no buffer for it, and DCP's strict load says so
    (a documented limit, readme "Checkpoints")."""
    from torch.distributed.checkpoint.api import CheckpointException

    with pytest.raises(CheckpointException, match="Missing key.*momentum_buffer"):
        _resume(0.0, 0.9, tmp_path)
