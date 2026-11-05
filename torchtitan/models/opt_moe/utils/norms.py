# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm
from torchtitan.models.base.module import Module


class SingleScaleRMSNorm(Module):
    """AKA SSNorm, from https://arxiv.org/abs/2506.19697."""

    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: tuple[int, ...]
    eps: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5

    def __init__(self, config: Config) -> None:
        super().__init__()
        normalized_shape = config.normalized_shape
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = config.eps

        self.ssnorm_scale = nn.Parameter(torch.empty((1,), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(input, self.normalized_shape, eps=self.eps)
        return self.ssnorm_scale * out

    def reset_parameters(self) -> None:
        nn.init.ones_(self.ssnorm_scale)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


# Every OPT MoE norm type maps onto a configurable upstream module, so norms are
# first-class ``Module``s: ``Module.parallelize`` can shard them and
# ``init_states`` initializes them, neither of which worked when they were bare
# ``nn.Module`` instances built by a lambda.
NORM_CONFIGS = {
    "layernorm": lambda dim, eps: LayerNorm.Config(normalized_shape=dim, eps=eps),
    "np_layernorm": lambda dim, eps: LayerNorm.Config(
        normalized_shape=dim, eps=eps, elementwise_affine=False
    ),
    "rmsnorm": lambda dim, eps: RMSNorm.Config(normalized_shape=dim, eps=eps),
    "np_rmsnorm": lambda dim, eps: RMSNorm.Config(
        normalized_shape=dim, eps=eps, elementwise_affine=False
    ),
    "ss_rmsnorm": lambda dim, eps: SingleScaleRMSNorm.Config(
        normalized_shape=dim, eps=eps
    ),
}


# Parameters each norm type owns. `np_*` variants are parameter-free, which is
# exactly why the sharding plan below can be shared: `Module._distribute_states`
# iterates the module's ACTUAL parameters and looks each one up, so naming a
# state a given norm does not have is inert, while failing to name one it does
# have leaves it a plain tensor and `fully_shard` rejects it under spmd_types.
NORM_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "layernorm": ("weight", "bias"),
    "np_layernorm": (),
    "rmsnorm": ("weight",),
    "np_rmsnorm": (),
    "ss_rmsnorm": ("ssnorm_scale",),
}

# Union of every norm parameter name. One plan covers all five types, so the
# sharding code does not have to branch on `norm_type` -- which it usually
# cannot see anyway, since the type is chosen per-module far from the plan.
ALL_NORM_PARAM_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(n for names in NORM_PARAM_NAMES.values() for n in names)
)


def norm_has_parameters(norm_type: str) -> bool:
    """Whether ``norm_type`` builds a norm with learnable parameters.

    The drift check lives here, not at module scope: a module-level ``assert``
    is stripped by ``python -O``, and on a real drift it would make the whole
    ``opt_moe`` package unimportable (including
    ``scripts/checkpoint_conversion/convert_to_hf.py``) rather than failing
    only for the norm type that is actually missing.
    """
    key = norm_type.lower()
    names = NORM_PARAM_NAMES.get(key)
    if names is None:
        if key in NORM_CONFIGS:
            raise KeyError(
                f"norm_type '{norm_type}' is in NORM_CONFIGS but missing from "
                "NORM_PARAM_NAMES; add its parameter names there so sharding "
                "plans can cover it."
            )
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")
    return bool(names)


def build_norm_config(
    norm_type: str, dim: int, eps: float = 1e-5, *, sharding_config=None
):
    """Return the ``Module.Config`` for the requested norm type.

    Args:
        norm_type: One of ``layernorm``, ``np_layernorm``, ``rmsnorm``,
            ``np_rmsnorm``, ``ss_rmsnorm``.
        dim: Normalized dimension.
        eps: Epsilon for numerical stability.
        sharding_config: Plan for the norm's parameters. Applied ONLY when
            ``norm_type`` actually has parameters -- the ``np_*`` variants own
            none, so stamping them would attach a plan that can never bind to
            anything and would needlessly take those modules off
            ``Module.parallelize``'s no-config fast path. Callers can therefore
            pass this unconditionally without knowing the norm type.

    Raises:
        NotImplementedError: If an unknown ``norm_type`` is provided.
    """
    norm_config_fn = NORM_CONFIGS.get(norm_type.lower())
    if norm_config_fn is None:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")
    cfg = norm_config_fn(dim, eps)
    if sharding_config is not None and norm_has_parameters(norm_type):
        cfg.sharding_config = sharding_config
    return cfg
