# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A single-process DiSCO written from the update rule, not from the optimizer.

Every parameter is held whole, in fp32, on one device. Per step and parameter:

    buf   <- b * buf + (1 - b) * g                 (momentum b, if 0 < b < 1)
    g_eff <- buf, or b * buf + (1 - b) * g if nesterov
    u     <- scale(m, n) * compute(g_eff)           (the group's compute and scale)
    w     <- w * (1 - wd * lr) - lr * u

Orthogonalisation calls the same zeropower kernel the optimizer uses (that is
the definition of the LMO, and this test is about the transport around it);
every row map and scale is transcribed from its formula below.
"""

import math

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


def decay_of(r: dict) -> float:
    """What weight decay scales W by per step."""
    wd = r["weight_decay"]
    return 1.0 - wd * r["lr"]


class ReferenceDiSCO:
    def __init__(self, params: dict[str, torch.Tensor], rules: dict[str, dict]):
        """`params`: name -> full fp32 tensor (cloned); `rules`: name -> group hyperparameters."""
        self.params = {k: v.detach().clone() for k, v in params.items()}
        self.rules = rules
        self.buf = {k: torch.zeros_like(v) for k, v in self.params.items()}
        self.last_update = {}
        self.t = 0

    @torch.no_grad()
    def lmo(self, g: torch.Tensor, r: dict) -> torch.Tensor:
        """Head-wise (`block_rows` / `block_cols`): every block of the unit is
        its own matrix, scaled by its own shape."""
        br, bc = r.get("block_rows", 0), r.get("block_cols", 0)
        if br or bc:
            parts = g.split(br, dim=0) if br else g.split(bc, dim=1)
            return torch.cat([self.matrix_lmo(b, r) for b in parts], dim=0 if br else 1)
        return self.matrix_lmo(g, r)

    def matrix_lmo(self, g: torch.Tensor, r: dict) -> torch.Tensor:
        if g.numel() == 1:
            return torch.sign(g)
        if r.get("rule_obj") is not None and r["rule_obj"].kind == "gram":
            return r["rule_obj"].reference(g).update  # the Gram rule, whole
        if r["compute"] in ("identity", "sign", "row-l2"):
            u = row_map(g, r["compute"], r["eps"])
        else:
            u = zeropower_backends[r["compute"]](g, steps=r["steps"], eps=r["eps"])
        return u * scale_factor(r["scale"], g.shape[-2], g.shape[-1])

    @torch.no_grad()
    def step(self, grads: dict[str, torch.Tensor]) -> None:
        self.t += 1
        for name, w in self.params.items():
            r = self.rules[name]
            g = grads[name].to(w.dtype)
            b = r["momentum"]
            if 0.0 < b < 1.0:
                buf = self.buf[name]
                buf.mul_(b).add_(g, alpha=1.0 - b)
                g_eff = buf if not r["nesterov"] else b * buf + (1.0 - b) * g
            else:
                g_eff = g
            if w.ndim < 2:  # scalars as they are, a vector as one [1, d] row
                u = self.lmo(g_eff if g_eff.numel() == 1 else g_eff.view(1, -1), r)
                u = u.view_as(g_eff)
                self.last_update[name] = u
                if r["weight_decay"] != 0.0:
                    w.mul_(decay_of(r))
                w.add_(u, alpha=-r["lr"])
                continue
            # Every declared unit is its own matrix: q / k / v of a fused wqkv,
            # w1 / w3 of a fused w13, one expert (or one expert's w1 / w3).
            rows, g_rows = w.view(-1, w.shape[-1]), g_eff.reshape(-1, w.shape[-1])
            upd = torch.zeros_like(rows)
            for k, (runs, shape) in enumerate(r["units"]):
                idx = torch.cat([torch.arange(s, s + n) for s, n in runs])
                W, G = rows[idx].view(shape), g_rows[idx].view(shape)
                u = self.unit_step((name, k), W, G, r)
                rows[idx] = W.reshape(len(idx), -1)
                if u is not None:
                    upd[idx] = u.reshape(len(idx), -1)
            self.last_update[name] = upd.view_as(w)

    def unit_step(self, key, w, g, r):
        """One unit's update, in place on `w`; returns the plain direction."""
        sphere = r.get("sphere")
        if sphere is not None:
            self._sphere_one(key, w, g, r, sphere)
            return None
        u = self.lmo(g, r)
        if r["weight_decay"] != 0.0:
            w.mul_(decay_of(r))
        w.add_(u, alpha=-r["lr"])
        return u

    def _sphere_one(self, name, w, g, r, c) -> None:
        """hyperball on one matrix, from its equation. The direction is the
        group's LMO without its scale."""
        W, M = w.float(), g.float()
        rr = W.norm()
        if c.tangent:
            M = M - (W * M).sum() / rr**2 * W
        U = self.lmo(M, {**r, "scale": "one"})
        U = U / U.norm()
        s = r["lr"]
        if c.exact_angle:
            cos = (W * U).sum() / rr
            s = math.tan(s) / ((1 - cos**2).sqrt() + cos * math.tan(s))
        B = W - s * rr * U
        w.copy_(rr * B / B.norm())


def rules_from_container(container, model) -> dict[str, dict]:
    """Per-parameter hyperparameters as the container's param groups resolved them."""
    from torchtitan.optimizers.disco.presets import FrobeniusAUS

    by_id = {}
    for opt in container.optimizers:
        rule_by_id = {id(e.param): e.rule for e in opt.entries}
        units_by_id = {
            id(e.param): [(u.rows.runs, tuple(u.shape)) for u in e.units]
            for e in opt.entries
        }
        sphere_by_id = {
            id(e.param): e.rule.compute
            for e in opt.entries
            if isinstance(e.rule.compute, FrobeniusAUS)
        }
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
                    "sphere": sphere_by_id.get(id(p)),
                    "rule_obj": rule_by_id.get(id(p)),
                    "units": units_by_id.get(id(p)),
                    "block_rows": int(a.get("block_rows", 0)),
                    "block_cols": int(a.get("block_cols", 0)),
                }
    return {n: by_id[id(p)] for n, p in model.named_parameters() if p.requires_grad}
