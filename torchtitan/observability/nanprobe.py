# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Env-gated NaN probe (debug copy only). Inert unless NAN_PROBE_FROM and
NAN_PROBE_DIR are set; then active on steps >= NAN_PROBE_FROM. Observes only:
every extra forward runs under no_grad, except the backward probe, which runs
only on a step that is already non-finite (the trainer's assert follows)."""

import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

FROM = int(os.environ.get("NAN_PROBE_FROM") or 0)
DIR = os.environ.get("NAN_PROBE_DIR") or ""
ON = FROM > 0 and bool(DIR)
BIG = float(os.environ.get("NAN_PROBE_BIG") or 1e4)
EAGER = os.environ.get("NAN_PROBE_EAGER", "1") == "1"
STOP = int(os.environ.get("NAN_PROBE_STOP") or 0)  # last step the replay runs
FORCE_BAD = (
    os.environ.get("NAN_PROBE_FORCE_BAD") == "1"
)  # smoke test of the bad-step path

_S: dict = {"step": 0, "cap": None, "mb": [], "pre": None}
_WRAP = ("_checkpoint_wrapped_module.", "_orig_mod.", "_fsdp_wrapped_module.")


def set_step(s: int) -> None:
    _S["step"] = int(s)


def should_stop(num_completed_steps: int) -> bool:
    """True once the step NAN_PROBE_STOP is done (the replay ends cleanly)."""
    return bool(ON and STOP and num_completed_steps >= STOP)


def on() -> bool:
    return ON and _S["step"] >= FROM


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _canon(n: str) -> str:
    for w in _WRAP:
        n = n.replace(w, "")
    return n


def _path(name: str) -> str:
    d = os.path.join(DIR, f"step{_S['step']}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.r{_rank():02d}")


def _dump(name: str, obj) -> None:
    with open(_path(name) + ".json", "w") as f:
        json.dump(obj, f, indent=1, default=float)


def log(msg: str) -> None:
    print(f"[nanprobe r{_rank()} s{_S['step']}] {msg}", flush=True)


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def _vec(t: torch.Tensor) -> torch.Tensor:
    """[n_nonfinite, max|finite|, rms(finite), numel, argmax row] of t."""
    x = _local(t).detach()
    dev = x.device
    if x.numel() == 0:
        return torch.zeros(5, device=dev)
    x = x.float()
    fin = torch.isfinite(x)
    xf = torch.where(fin, x, torch.zeros_like(x)).abs()
    rows = xf.reshape(xf.shape[0], -1).amax(dim=-1) if xf.ndim >= 2 else xf
    return torch.stack(
        [
            (~fin).sum().float(),
            xf.max(),
            (xf.square().sum() / x.numel()).sqrt(),
            torch.tensor(float(x.numel()), device=dev),
            rows.argmax().float(),
        ]
    )


def _table(named) -> list[dict]:
    if not named:
        return []
    T = torch.stack([_vec(t) for _, t in named]).cpu().tolist()
    return [
        {
            "name": n,
            "nonfinite": int(r[0]),
            "maxabs": r[1],
            "rms": r[2],
            "numel": int(r[3]),
            "argmax_row": int(r[4]),
        }
        for (n, _), r in zip(named, T)
    ]


# ---------------------------------------------------------------- trainer hooks
def capture(inputs, labels, model_kwargs, loss_kwargs) -> None:
    _S["cap"] = (inputs, labels, model_kwargs, loss_kwargs)


def on_microbatch(engine, microbatch, detached_loss, accumulation_index: int) -> None:
    if accumulation_index == 0:
        _S["mb"] = []
    d = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in microbatch.as_input_dict().items()
    }
    d["num_valid_tokens"] = int(microbatch.num_valid_tokens)
    torch.save(d, _path(f"mb{accumulation_index}") + ".pt")
    lv = float(detached_loss.float().item())
    _S["mb"].append(
        {
            "microbatch": accumulation_index,
            "num_valid_tokens": int(microbatch.num_valid_tokens),
            "loss_local_over_global_tokens": lv,
            "loss_finite": bool(torch.isfinite(torch.tensor(lv))),
        }
    )


def _params(engine):
    return [(_canon(n), p) for m in engine.model_parts for n, p in m.named_parameters()]


def after_backward(engine, grad_norm: torch.Tensor) -> None:
    t0 = time.time()
    params = _params(engine)
    gn = float(grad_norm.float().item())
    loss_ok = bool(engine.loss_is_finite.item())
    step_ok = loss_ok and bool(torch.isfinite(torch.tensor(gn))) and not FORCE_BAD
    grads = _table([(n, p.grad) for n, p in params if p.grad is not None])
    weights = _table([(n, p) for n, p in params])
    _dump(
        "step",
        {
            "step": _S["step"],
            "rank": _rank(),
            "microbatches": _S["mb"],
            "grad_norm_global": gn,
            "loss_is_finite_all_ranks": loss_ok,
            "step_finite": step_ok,
            "grads_nonfinite": [g["name"] for g in grads if g["nonfinite"]],
            "grads": grads,
            "weights_pre_update": weights,
        },
    )
    bad = [g["name"] for g in grads if g["nonfinite"]]
    log(
        f"grad_norm={gn!r} loss_finite_all={loss_ok} local_mb={_S['mb']} n_param_grads_nonfinite={len(bad)} first={bad[:3]}"
    )
    _S["pre"] = [(n, _local(p).detach().clone()) for n, p in params]
    try:
        forward_probe(engine, "compiled")
        if EAGER:
            forward_probe(engine, "eager")
    except Exception as ex:  # report, never alter training
        log(f"forward probe failed: {type(ex).__name__}: {ex}")
    if not step_ok:
        try:
            save_weights(engine)
        except Exception as ex:
            log(f"save_weights failed: {type(ex).__name__}: {ex}")
        try:
            backward_probe(engine)
        except Exception as ex:
            log(f"backward probe failed: {type(ex).__name__}: {ex}")
        log("step is non-finite; probe done, the trainer's assert follows")
    log(f"after_backward probe took {time.time() - t0:.1f}s")


def after_update(engine) -> None:
    pre = dict(_S["pre"] or [])
    named, deltas = [], []
    for n, p in _params(engine):
        named.append((n, p))
        if n in pre:
            deltas.append((n, _local(p).detach().float() - pre[n].float()))
    _dump("update", {"weights_post_update": _table(named), "delta": _table(deltas)})
    _S["pre"] = None


# ---------------------------------------------------------------- forward probe
def _out_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor):
        return o[0]
    return None


def _fsdp_reshard_all(model) -> None:
    from torch.distributed.fsdp import FSDPModule

    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


def _ctx(engine):
    from torchtitan.distributed import utils as dist_utils

    return engine.parallelism_context.activate_spmd(
        typechecking=engine.config.debug.spmd_typechecking,
    )


def _patch_layers(model, wrap):
    saved = {}
    for name, layer in model.layers.items():
        orig = layer._compiled_call_impl
        saved[name] = orig
        layer._compiled_call_impl = wrap(
            name, layer, orig if orig is not None else layer._call_impl
        )
    return saved


def _unpatch_layers(model, saved) -> None:
    for name, layer in model.layers.items():
        layer._compiled_call_impl = saved[name]


def forward_probe(engine, mode: str) -> None:
    """no_grad forward of this rank's microbatch. "compiled": the compiled
    blocks, stats of each block's output. "eager": each block's uncompiled
    _call_impl with forward hooks on every submodule, in call order."""
    if _S["cap"] is None:
        return
    inputs, labels, mk, lk = _S["cap"]
    model = engine.model_parts[0]
    rec: list = []  # (name, tensor-stat vector) in call order
    hooks = []

    def wrap(name, layer, orig):
        def f(*a, **k):
            out = orig(*a, **k) if mode == "compiled" else layer._call_impl(*a, **k)
            rec.append((f"layers.{name}", _vec(out)))
            return out

        return f

    def fhook(name):
        def h(mod, args, out):
            t = _out_tensor(out)
            if t is not None:
                rec.append((name, _vec(t)))

        return h

    if mode == "eager":
        for n, m in model.named_modules():
            if n.startswith("layers.") and n.count(".") >= 2:
                hooks.append(m.register_forward_hook(fhook(_canon(n))))
    for n in ("tok_embeddings", "norm"):
        m = getattr(model, n, None)
        if m is not None:
            hooks.append(m.register_forward_hook(fhook(n)))
    saved = _patch_layers(model, wrap)
    rc = getattr(model, "residual_capture", None)
    if hasattr(model, "residual_capture"):
        model.residual_capture = None
    try:
        with torch.no_grad(), _ctx(engine):
            pred = model(inputs, **mk)
            tok = per_token_loss(engine, pred, labels) if mode == "compiled" else None
    finally:
        for h in hooks:
            h.remove()
        _unpatch_layers(model, saved)
        if hasattr(model, "residual_capture"):
            model.residual_capture = rc
        _fsdp_reshard_all(model)
    names = [r[0] for r in rec]
    T = torch.stack([r[1] for r in rec]).cpu().tolist() if rec else []
    acts = [
        {
            "name": n,
            "nonfinite": int(r[0]),
            "maxabs": r[1],
            "rms": r[2],
            "argmax_row": int(r[4]),
        }
        for n, r in zip(names, T)
    ]
    first = next((a for a in acts if a["nonfinite"] or a["maxabs"] > BIG), None)
    out = {"mode": mode, "first_nonfinite_or_big": first, "acts": acts}
    if tok is not None:
        out["tokens"] = doc_stats(tok, labels, mk, engine)
    _dump(f"fwd_{mode}", out)
    log(f"fwd_{mode}: first non-finite/|x|>{BIG:g}: {first}")
    if tok is not None:
        s = out["tokens"]
        log(
            f"fwd_compiled: loss_local={s['loss_local_over_global_tokens']!r} max_tok_ce={s['max_token_ce']!r} n_nonfinite_ce={s['n_nonfinite_ce']}"  # noqa: B950
        )


def per_token_loss(engine, pred, labels) -> torch.Tensor:
    from torch.distributed.fsdp import FSDPModule

    lf = engine.loss_fn
    lm_head = getattr(lf, "lm_head", None)
    n_chunks = int(getattr(lf, "num_chunks", 8))
    h = _local(pred)
    lab = _local(labels)
    if lm_head is None:
        return F.cross_entropy(h.float(), lab, reduction="none", ignore_index=-100)
    fsdp = isinstance(lm_head, FSDPModule)
    if fsdp:
        lm_head.set_reshard_after_forward(False)
        lm_head.unshard()
    try:
        outs = []
        for hc, lc in zip(h.chunk(n_chunks, dim=0), lab.chunk(n_chunks, dim=0)):
            outs.append(
                F.cross_entropy(
                    lm_head(hc).float(), lc, reduction="none", ignore_index=-100
                )
            )
    finally:
        if fsdp:
            lm_head.set_reshard_after_forward(True)
            lm_head.reshard()
    return torch.cat(outs)


def doc_stats(ce: torch.Tensor, labels, mk, engine) -> dict:
    lab = _local(labels)
    pos = mk.get("positions")
    pos = _local(pos) if pos is not None else None
    valid = lab != -100
    T = ce.shape[0]
    if pos is None:
        doc = torch.zeros(T, dtype=torch.long, device=ce.device)
    else:
        doc = torch.cumsum((pos == 0).long(), 0) - 1
        doc = doc.clamp_min(0)
    nd = int(doc.max().item()) + 1
    cev = torch.where(valid, ce, torch.zeros_like(ce))
    fin = torch.isfinite(cev)
    cef = torch.where(fin, cev, torch.zeros_like(cev))
    s = torch.zeros(nd, device=ce.device).scatter_add_(0, doc, cef)
    c = torch.zeros(nd, device=ce.device).scatter_add_(0, doc, valid.float())
    mx = torch.zeros(nd, device=ce.device).scatter_reduce_(0, doc, cef, "amax")
    nf = torch.zeros(nd, device=ce.device).scatter_add_(0, doc, (~fin).float())
    ln = torch.zeros(nd, device=ce.device).scatter_add_(0, doc, torch.ones_like(cef))
    start = torch.full((nd,), T, device=ce.device, dtype=torch.long).scatter_reduce_(
        0, doc, torch.arange(T, device=ce.device), "amin"
    )
    gvt = _S["cap"][3].get("global_valid_tokens") if _S["cap"] else None
    gvt = float(gvt) if gvt is not None else None
    order = torch.argsort(mx, descending=True)[:8]
    top = [
        {
            "doc": int(i),
            "start": int(start[i]),
            "len": int(ln[i]),
            "valid": int(c[i]),
            "mean_ce": float(s[i] / c[i].clamp_min(1)),
            "max_ce": float(mx[i]),
            "n_nonfinite": int(nf[i]),
        }
        for i in order.tolist()
    ]
    bad = torch.nonzero(~fin).flatten()[:16].tolist()
    am = int(cef.argmax())
    return {
        "n_tokens": T,
        "n_valid": int(valid.sum()),
        "n_docs": nd,
        "doc_len_min": int(ln.min()),
        "doc_len_max": int(ln.max()),
        "n_docs_len_le_2": int((ln <= 2).sum()),
        "loss_sum_finite": float(cef.sum()),
        "loss_local_over_global_tokens": (float(cef.sum()) / gvt) if gvt else None,
        "max_token_ce": float(cef.max()),
        "max_token_ce_at": {"token": am, "doc": int(doc[am])},
        "n_nonfinite_ce": int((~fin).sum()),
        "nonfinite_ce_tokens": [
            {"token": i, "doc": int(doc[i]), "offset": int(i - start[doc[i]])}
            for i in bad
        ],
        "top_docs_by_max_ce": top,
    }


# ---------------------------------------------------------------- backward probe
def backward_probe(engine) -> None:
    """Only on a non-finite step: the compiled forward/backward again, with
    hooks on each block output's gradient (local, before any reduction)."""
    if _S["cap"] is None:
        return
    inputs, labels, mk, lk = _S["cap"]
    model = engine.model_parts[0]
    rec: list = []

    def gh(name):
        def h(g):
            rec.append((name, _vec(g)))

        return h

    def wrap(name, layer, orig):
        def f(*a, **k):
            out = orig(*a, **k)
            if out.requires_grad:
                out.register_hook(gh(f"grad_out.layers.{name}"))
            return out

        return f

    hooks = []
    for n in ("tok_embeddings", "norm"):
        m = getattr(model, n, None)
        if m is not None:

            def fh(mod, args, out, n=n):
                t = _out_tensor(out)
                if t is not None and t.requires_grad:
                    t.register_hook(gh(f"grad_out.{n}"))

            hooks.append(m.register_forward_hook(fh))
    saved = _patch_layers(model, wrap)
    rc = getattr(model, "residual_capture", None)
    if hasattr(model, "residual_capture"):
        model.residual_capture = None
    try:
        with _ctx(engine):
            pred = model(inputs, **mk)
            loss, _ = engine.loss_fn(pred, labels, **lk)
            del pred
            loss.backward()
    finally:
        for h in hooks:
            h.remove()
        _unpatch_layers(model, saved)
        if hasattr(model, "residual_capture"):
            model.residual_capture = rc
    names = [r[0] for r in rec]
    T = torch.stack([r[1] for r in rec]).cpu().tolist() if rec else []
    acts = [
        {
            "name": n,
            "nonfinite": int(r[0]),
            "maxabs": r[1],
            "rms": r[2],
            "argmax_row": int(r[4]),
        }
        for n, r in zip(names, T)
    ]
    first = next((a for a in acts if a["nonfinite"]), None)
    lv = float(loss.detach().float().item())
    _dump(
        "bwd",
        {
            "loss_local_rerun": lv,
            "first_nonfinite_in_backward_order": first,
            "grads_in_backward_order": acts,
        },
    )
    log(
        f"bwd: rerun loss_local={lv!r} first non-finite output-grad (backward order): {first}"
    )


def save_weights(engine) -> None:
    sd = {}
    for n, p in _params(engine):
        t = p.detach()
        if isinstance(t, DTensor):
            t = t.full_tensor()
        if _rank() == 0:
            sd[n] = t.cpu()
    if _rank() == 0:
        torch.save(sd, os.path.join(DIR, f"step{_S['step']}", "weights_pre_update.pt"))
        log(f"saved {len(sd)} full weights")


# ---------------------------------------------------------------- optimizer hooks
def disco_tensors(opt, stage: str, t: dict) -> None:
    named, units = [], {}
    for e in opt.entries:
        if e.i in t:
            named.append((e.name, t[e.i]))
    tab = _table(named)
    for row, e in zip(tab, [e for e in opt.entries if e.i in t]):
        if row["nonfinite"]:
            try:
                from torchtitan.optimizers.disco.optimizer import unit_rows_local

                x = e.rows2d(t[e.i]) if t[e.i].ndim != 2 else t[e.i]
                per = []
                for k, runs in enumerate(e.unit_local):
                    if runs.n_rows:
                        r = unit_rows_local(x, runs)
                        per.append(
                            {
                                "unit": k,
                                "nonfinite": int((~torch.isfinite(r)).sum()),
                                "rows": int(r.shape[0]),
                                "min_row_norm": float(
                                    r.float().nan_to_num(0).norm(dim=-1).min()
                                ),
                            }
                        )
                units[e.name] = per
            except Exception as ex:
                units[e.name] = f"{type(ex).__name__}: {ex}"
    _dump(f"disco_{stage}", {"entries": tab, "nonfinite_units": units})
    bad = [r["name"] for r in tab if r["nonfinite"]]
    if bad:
        log(f"disco {stage}: non-finite in {bad[:5]} (+{max(0, len(bad) - 5)})")


def disco_run(opt, rule, st, w, state, res, pairs) -> None:
    out = {}
    for k in ("update", "weight", "state", "alpha"):
        v = getattr(res, k, None)
        if isinstance(v, torch.Tensor):
            out[k] = v
    flag = any(bool((~torch.isfinite(v)).any()) for v in out.values())
    names = [f"{ee.name}#{getattr(un, 'pos', '?')}" for ee, un in pairs]
    x = st.float()
    xs = x.reshape(x.shape[0], -1) if x.ndim == 3 else x.reshape(1, -1)
    rec = {
        "units": names,
        "in_norm": xs.norm(dim=-1).tolist(),
        "in_min_row_norm": (
            x.norm(dim=-1).amin(dim=-1).tolist() if x.ndim == 3 else None
        ),
        "nonfinite_out": flag,
        "out": {k: _vec(v).cpu().tolist() for k, v in out.items()},
    }
    if w is not None:
        wf = w.float()
        rec["w_min_row_norm"] = (
            wf.norm(dim=-1).amin(dim=-1).tolist() if wf.ndim == 3 else None
        )
    lst = _S.setdefault("runs", {}).setdefault(_S["step"], [])
    lst.append(rec)
    if flag:
        torch.save(
            {
                "st": st.cpu(),
                "w": None if w is None else w.cpu(),
                "state": None if state is None else state.cpu(),
                "units": names,
                "ctx": tuple(opt._ctx(pairs[0][0])),
            },
            _path(f"disco_run_bad_{len(lst)}") + ".pt",
        )
        log(f"disco _run non-finite output for {names[:4]}")


def disco_flush() -> None:
    runs = _S.get("runs", {}).pop(_S["step"], None)
    if runs is not None:
        _dump("disco_runs", runs)
