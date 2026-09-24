# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A single-process DiSCO written from the update rule, not from the optimizer.

Every parameter is held whole, in fp32, on one device. Per step and parameter:

    buf   <- (1 - m) * buf + m * g                 (momentum, if 0 < m < 1)
    g_eff <- buf, or lerp(buf, g, m) if nesterov
    u     <- scale(m, n) * compute(g_eff)           (the group's compute and scale)
    w     <- w * (1 - wd * lr) - lr * u

Orthogonalisation calls the same zeropower kernel the optimizer uses (that is
the definition of the LMO, and this test is about the transport around it);
every row map and scale is transcribed from its formula below.
"""

import torch

from torchtitan.optimizers.kernels.zeropower import zeropower_backends


def row_map(u: torch.Tensor, compute: str, eps: float) -> torch.Tensor:
    if compute == "identity":
        return u
    if compute == "sign":
        return torch.sign(u)
    if compute == "row-l2":
        row = u.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
        return (u.float() / (row + eps)).to(u.dtype)
    raise ValueError(f"reference has no formula for compute={compute!r}")


def scale_factor(scale: str, m: int, n: int) -> float:
    return {
        "one": 1.0,
        "spectral": (m / n) ** 0.5,
        "sqrt_n": n**0.5,
        "inv_sqrt_n": n**-0.5,
        "n": float(n),
        "inv_n": 1.0 / n,
    }[scale]


class ReferenceDiSCO:
    def __init__(self, params: dict[str, torch.Tensor], rules: dict[str, dict]):
        """`params`: name -> full fp32 tensor (cloned); `rules`: name -> group hyperparameters."""
        self.params = {k: v.detach().clone() for k, v in params.items()}
        self.rules = rules
        self.buf = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_update = {}
        self.aus_v = {}

    @torch.no_grad()
    def lmo(self, g: torch.Tensor, r: dict) -> torch.Tensor:
        if g.numel() == 1:
            return torch.sign(g)
        if r.get("rule_obj") is not None and r["rule_obj"].kind == "gram":
            return r["rule_obj"].reference(g).update  # the Gram rule, whole
        if r["compute"] in ("identity", "sign", "row-l2"):
            u = row_map(g, r["compute"], r["eps"])
        else:
            fn = zeropower_backends[r["compute"]]
            if g.ndim == 3:
                u = torch.stack(
                    [
                        fn(g[e], steps=r["steps"], eps=r["eps"])
                        for e in range(g.shape[0])
                    ]
                )
            else:
                u = fn(g, steps=r["steps"], eps=r["eps"])
        return u * scale_factor(r["scale"], g.shape[-2], g.shape[-1])

    @torch.no_grad()
    def step(self, grads: dict[str, torch.Tensor]) -> None:
        for name, w in self.params.items():
            r = self.rules[name]
            g = grads[name].to(w.dtype)
            m = r["momentum"]
            if 0.0 < m < 1.0:
                buf = self.buf[name]
                buf.mul_(1.0 - m).add_(g, alpha=m)
                g_eff = buf if not r["nesterov"] else torch.lerp(buf, g, m)
            else:
                g_eff = g
            u = self.lmo(g_eff, r)
            self.last_update[name] = u
            if r.get("aus"):
                if w.ndim == 2:
                    self.aus_step(name, w, u, r)
                else:  # experts: every expert its own matrix and iterate
                    for e in range(w.shape[0]):
                        self.aus_step((name, e), w[e], u[e], r)
                continue
            if r["weight_decay"] != 0.0:
                w.mul_(1.0 - r["weight_decay"] * r["lr"])
            w.add_(u, alpha=-r["lr"])

    @torch.no_grad()
    def aus_step(self, name, w, u, r) -> None:
        """The AUS map from its equations, per matrix, with the same warm
        power iteration the rule uses (`sigma_max_warm`), iterate kept per name."""
        from torchtitan.optimizers.helpers.power_iteration import sigma_max_warm

        eta, iters = r["aus_eta"], r["aus_sigma_iters"]
        tiny = torch.finfo(torch.float32).tiny
        W = w.float() * (1.0 - r["weight_decay"] * r["lr"])
        U = u.float()
        v = self.aus_v.get(name)
        R, v = sigma_max_warm(W[None], None if v is None else v[None], iters)
        sU, _ = sigma_max_warm(U[None], v, 1)
        B = (
            W[None]
            - (eta * R)[:, None, None] * U[None] / sU.clamp_min(tiny)[:, None, None]
        )
        sB, _ = sigma_max_warm(B, v, iters)
        R_new, _ = sigma_max_warm(W[None] - eta * U[None], v, iters)
        self.aus_v[name] = v[0]
        w.copy_((B * (R_new / sB.clamp_min(tiny))[:, None, None])[0].to(w.dtype))


def rules_from_container(container, model) -> dict[str, dict]:
    """Per-parameter hyperparameters as the container's param groups resolved them."""
    from torchtitan.optimizers.disco.presets import AUSMap

    by_id = {}
    for opt in container.optimizers:
        # The rule an entry actually runs (an `aus` group with row args is a row rule).
        aus_by_id = {
            id(e.param): isinstance(e.rule.compute, AUSMap) for e in opt.entries
        }
        rule_by_id = {id(e.param): e.rule for e in opt.entries}
        for group in opt.param_groups:
            for p in group["params"]:
                a = group["args"]
                by_id[id(p)] = {
                    "lr": group["lr"],
                    "momentum": group["momentum"],
                    "nesterov": group["nesterov"],
                    "weight_decay": group["weight_decay"],
                    "eps": 1e-30,
                    "scale": a.get("scale", "one"),
                    "compute": a.get("compute", "identity"),
                    "steps": a.get("steps", 0),
                    "aus": aus_by_id.get(id(p), False),
                    "rule_obj": rule_by_id.get(id(p)),
                    "aus_sigma_iters": int(a.get("sigma_iters", 8)),
                    "aus_eta": 0.0,  # set per step by the test from the rule's schedule
                }
    return {n: by_id[id(p)] for n, p in model.named_parameters() if p.requires_grad}
