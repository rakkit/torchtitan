# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat
from typing import cast, List, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.optim.optimizer import _device_dtype_check_for_fused, _get_scalar_dtype

from .newton_schulz_triton import newton_schulz_triton, polar_express_triton

# def zeropower_via_svd(G, **kwargs):
#     U, S, V = G.svd()
#     X = U @ V.T
#     return X


def zeropower_via_svd(G, **kwargs):
    original_dtype = G.dtype
    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)  # batch of 1

    assert G.dim() == 3, f"Expected 2D or 3D tensor, got {G.shape}"

    G32 = G.to(torch.float32)  # SVD does not support bfloat16 reliably
    rows, cols = G32.shape[-2], G32.shape[-1]
    transposed = rows > cols
    if transposed:
        G32 = G32.transpose(-2, -1)

    X_list = []
    for b in range(G32.shape[0]):
        U, S, V = G32[b].svd()  # per-matrix SVD (matches your API choice)
        Xb = U @ V.t()  # orthogonal factor
        X_list.append(Xb)

    X = torch.stack(X_list, dim=0)

    if transposed:
        X = X.transpose(-2, -1)
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype).contiguous()


# Polar Express
# Polar-express coefficients (https://arxiv.org/abs/2505.16932), with the
# 1/1.01 stabiliser applied to every triple except the limit. Factored out of
# `zeropower_via_polar_express` so `gram_newton_schulz.gram_polar_express`
# iterates on literally the same numbers -- the two backends should differ only
# in how the polynomial is evaluated, never in which polynomial it is.
_POLAR_EXPRESS_BASE = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.948690853482295, -2.908902115962949, 0.5518191394370137),
    (3.318419657370602, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.668903984574749, 0.4188073119525673),
    (1.891301407787398, -1.267995827194587, 0.3768040894852483),
    (1.875001480853448, -1.250001645399949, 0.3750001645474248),
    (1.875000000000000, -1.250000000000000, 0.375000000000000),  # limit
]
_POLAR_EXPRESS_STABILISED = [
    (a / 1.01, b / (1.01**3), c / (1.01**5))
    for (a, b, c) in _POLAR_EXPRESS_BASE[:-1]
] + [_POLAR_EXPRESS_BASE[-1]]


def polar_express_coefficients(steps: int) -> list[tuple[float, float, float]]:
    """First `steps` polar-express triples, repeating the limit triple if
    `steps` exceeds the table."""
    coeffs = _POLAR_EXPRESS_STABILISED + list(
        repeat(
            _POLAR_EXPRESS_STABILISED[-1],
            max(0, steps - len(_POLAR_EXPRESS_STABILISED)),
        )
    )
    return coeffs[:steps]


@torch.compile
def zeropower_via_polar_express(G, steps=5, eps=1e-7):
    # https://arxiv.org/abs/2505.16932
    # Support 2D (unsqueezed to batch=1) and 3D inputs.
    is_2d = G.dim() == 2
    if is_2d:
        G = G.unsqueeze(0)
    assert (
        G.dim() == 3
    ), f"Please make sure gradients are 2D or 3D tensors, got shape: {G.shape}"

    coeffs = polar_express_coefficients(steps)

    original_dtype = G.dtype
    X = G.bfloat16()

    # Transpose whole batch if rows > cols (same rule as your 2D version).
    rows, cols = X.shape[-2], X.shape[-1]
    transposed = False
    if rows > cols:
        X = X.transpose(1, 2)
        transposed = True

    # Per-matrix normalisation; matches your scalar normalisation when batch=1.
    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = X / (norm * 1.01 + eps)  # ensure top singular value <= 1

    # Main loop (batched matmuls).
    for k in range(steps):
        a, b, c = coeffs[k]
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)

    if transposed:
        X = X.transpose(1, 2)
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype)


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert (
        len(G.shape) == 2 or len(G.shape) == 3
    ), f"Please make sure gradients are 2D tensors to use NS, got shape: {G.shape}"

    is_2d = len(G.shape) == 2
    if is_2d:
        G = G.unsqueeze(0)

    assert (
        len(G.shape) == 3
    ), f"Please make sure gradients are 2D or 3D tensors, got shape: {G.shape}"

    a, b, c = (3.4445, -4.7750, 2.0315)
    original_dtype = G.dtype
    X = G.bfloat16()

    # Determine if we need to transpose the matrices based on their dimensions
    rows, cols = X.shape[-2], X.shape[-1]
    transposed = False
    if rows > cols:
        transposed = True
        X = X.transpose(1, 2)

    # Normalize each matrix in the batch individually
    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = X / (norm + eps)  # ensure top singular value <= 1 for each matrix

    # Perform the iteration using batched matrix multiplication (torch.bmm)
    for _ in range(steps):
        # A = X @ X.T for a batch
        A = torch.bmm(X, X.transpose(1, 2))
        # B = b*A + c*A@A for a batch
        B = b * A + c * torch.bmm(A, A)
        # X = a*X + B@X for a batch
        X = a * X + torch.bmm(B, X)

    # Transpose back if we did it at the beginning
    if transposed:
        X = X.transpose(1, 2)

    # If the original input was a 2D tensor, squeeze the batch dimension out
    if is_2d:
        X = X.squeeze(0)

    return X.to(original_dtype)


@torch.compile(dynamic=False, fullgraph=True)
def hybrid_polar_express_triton_gate_up(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
):
    dtype = G.dtype
    G_fp32 = G.float()
    norm = G_fp32.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    G = (G_fp32 / norm).to(dtype)
    return polar_express_triton(G, steps, eps)


@torch.compile(dynamic=False, fullgraph=True)
def hybrid_polar_express_triton_down(
    G: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    # nn.Linear layout: [D_out, D_in]
    # w2: [D_model, D_ff]
    # Compute one FP32 norm per input column.
    original_dtype = G.dtype
    G_fp32 = G.float()

    column_norm = G_fp32.norm(
        p=2,
        dim=0,
        keepdim=True,
    ).clamp_min(1e-8)

    G_normalized = (G_fp32 / column_norm).to(original_dtype)

    return polar_express_triton(G_normalized, steps, eps)


zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    polar_express=zeropower_via_polar_express,
    newtonschulz5_triton=newton_schulz_triton,
    polar_express_triton=polar_express_triton,
    hybrid_polar_express_triton_gate_up=hybrid_polar_express_triton_gate_up,
    hybrid_polar_express_triton_down=hybrid_polar_express_triton_down,
    identity=lambda x, **kwargs: x,
)

# Registered after the dict so `gram_newton_schulz` can import
# `polar_express_coefficients` from this module without a cycle. Opt in with
# `--optimizer.zeropower_backend gram_polar_express`; the default is unchanged.
# This one is on the *update path*, so switching it changes the training
# trajectory -- see that module's docstring for the measured speed and accuracy.
try:
    from .gram_newton_schulz import gram_polar_express as _gram_polar_express

    zeropower_backends["gram_polar_express"] = _gram_polar_express
except Exception:  # pragma: no cover - optional, never break import
    pass


def _init_adamw_group(
    self,
    group,
    params_with_grad,
    grads,
    amsgrad,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
):
    has_complex = False
    for p in group["params"]:
        if p.grad is None and not self.state[p]["use_muon"]:
            continue
        has_complex |= torch.is_complex(p)
        params_with_grad.append(p)
        if p.grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if "exp_avg" not in state:
            if group["adamw_fused"]:
                _device_dtype_check_for_fused(p)
            # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
            # This is because kernel launches are costly on CUDA and XLA.
            state["step"] = (
                torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(is_fused=group["adamw_fused"]),
                    device=p.device,
                )
                if group["adamw_capturable"] or group["adamw_fused"]
                else torch.tensor(0.0, dtype=_get_scalar_dtype())
            )
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if group["adamw_amsgrad"]:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])
        if group["adamw_differentiable"] and state["step"].requires_grad:
            raise RuntimeError(
                "`requires_grad` is not supported for `step` in differentiable mode"
            )

        # Foreach without capturable does not support a tensor lr
        if (
            group["adamw_foreach"]
            and isinstance(group["adamw_lr"], Tensor)
            and not group["adamw_capturable"]
        ):
            raise RuntimeError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )

        state_steps.append(state["step"])
    return has_complex


@torch.no_grad()
def update_adamw(self):
    """
    Optimized AdamW implementation using PyTorch's _functional.adamw or foreach operations.
    """
    for group in self.param_groups:
        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_sqs: List[Tensor] = []
        max_exp_avg_sqs: List[Tensor] = []
        state_steps: List[Tensor] = []
        amsgrad: bool = group["adamw_amsgrad"]
        beta1, beta2 = cast(Tuple[float, float], group["adamw_betas"])

        has_complex = _init_adamw_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        torch.optim._functional.adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["adamw_lr"],
            weight_decay=group["adamw_weight_decay"],
            eps=group["adamw_eps"],
            maximize=group["adamw_maximize"],
            foreach=group["adamw_foreach"],
            capturable=group["adamw_capturable"],
            differentiable=group["adamw_differentiable"],
            fused=group["adamw_fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            has_complex=has_complex,
        )


def gather_full_grad(g):
    """Gathers the full gradient across all distributed processes using DTensor."""
    assert isinstance(g, DTensor), "Expected gradient to be a DTensor"
    replicated_grad = g.redistribute(
        placements=[Replicate()] * g.device_mesh.ndim
    )  # make sure all rank has the same shape
    return replicated_grad


def shard_full_grad(g):
    """Extracts the correct shard for the current rank from a replicated DTensor."""
    assert isinstance(g, DTensor), "Expected gradient to be a DTensor"
    return g.redistribute(placements=[Shard(0)])
