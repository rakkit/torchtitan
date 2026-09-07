# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static checks on disco.py (plan Verification steps 6-7)."""
import ast, re, sys

SRC = "resources/torchtitan/torchtitan/optimizers/disco.py"
src = open(SRC).read()
tree = ast.parse(src)
fails = []


def check(name, cond, detail=""):
    print(
        ("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else "")
    )
    if not cond:
        fails.append(name)


fns = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        fns[node.name] = node


def calls_in(fn, name):
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            fname = getattr(f, "id", None) or getattr(f, "attr", None)
            if fname == name:
                out.append(n.lineno)
    return sorted(out)


print("== ordering: norm < radial < gram in every step_* ==")
for step in ["step_embedding", "step_experts", "step_ddp", "step_fsdp"]:
    fn = fns[step]
    # step_experts computes its norms up front via _batched_expert_norms
    # (which calls calculate_norm_batched); the others call calculate_norm
    # inline. Either counts as "the norm pass" for ordering purposes.
    norm = (
        calls_in(fn, "calculate_norm")
        + calls_in(fn, "calculate_norm_batched")
        + calls_in(fn, "_batched_expert_norms")
    )
    rad = calls_in(fn, "calculate_radial_metrics")
    gram = calls_in(fn, "calculate_gram_metrics")
    check(
        f"{step}: has all three",
        bool(norm and rad and gram),
        f"norm={norm} radial={rad} gram={gram}",
    )
    if norm and rad and gram:
        check(
            f"{step}: norm before radial",
            min(norm) < min(rad),
            f"{min(norm)} < {min(rad)}",
        )
        check(
            f"{step}: radial before gram",
            max(rad) < min(gram),
            f"{max(rad)} < {min(gram)}",
        )

print("\n== single source of truth: one radial call per step_* ==")
for step in ["step_embedding", "step_experts", "step_ddp", "step_fsdp"]:
    n = len(calls_in(fns[step], "calculate_radial_metrics"))
    check(f"{step}: exactly one calculate_radial_metrics", n == 1, f"count={n}")

print("\n== every radial call is inside a need_to_calculate_norm gate ==")
lines = src.split("\n")
for step in ["step_embedding", "step_experts", "step_ddp", "step_fsdp"]:
    fn = fns[step]
    ln = calls_in(fn, "calculate_radial_metrics")[0]
    # walk upward for an enclosing `if ... need_to_calculate_norm ...`
    indent = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
    found = False
    for j in range(ln - 2, fn.lineno - 2, -1):
        L = lines[j]
        if not L.strip():
            continue
        ind = len(L) - len(L.lstrip())
        if ind < indent and L.lstrip().startswith(("if ", "elif ")):
            indent = ind
            if "need_to_calculate_norm" in L:
                found = True
                break
            # step_ddp gates indirectly: its radial call sits under
            # `if radial_local_flat is not None:`, and that buffer is
            # initialised to None and only ever assigned inside a
            # `if need_to_calculate_norm:` block. Accept the sentinel, but
            # verify that property rather than assuming it.
            m = re.match(r"if (\w+) is not None:", L.strip())
            if m:
                guard = m.group(1)
                decl = re.search(rf"^\s*{guard} = None$", src, re.M)
                assigns = [
                    mm.start()
                    for mm in re.finditer(rf"^\s+{guard} = (?!None$)", src, re.M)
                ]
                gate = re.search(r"^\s*if need_to_calculate_norm:$", src, re.M)
                if decl and gate and assigns and all(a > gate.start() for a in assigns):
                    found = True
                    break
    check(f"{step}: radial gated on need_to_calculate_norm", found, f"line {ln}")

print("\n== every radial call passes spectral= ==")
for step in ["step_embedding", "step_experts", "step_ddp", "step_fsdp"]:
    fn = fns[step]
    node = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        == "calculate_radial_metrics"
    ][0]
    kw = {k.arg for k in node.keywords}
    check(f"{step}: spectral= supplied", "spectral" in kw, str(sorted(kw)))

print("\n== packing order unchanged (segment names + order) ==")
seg_orders = re.findall(r"_pack_segments\(\s*\[(.*?)\]\s*[,)]", src, re.S)
for i, blk in enumerate(seg_orders):
    names = re.findall(r'\(\s*"([a-z_]+)"', blk)
    print(f"    _pack_segments #{i}: {names}")
expected = ["upd", "w", "gram", "gram_vec", "radial", "upd_spec", "w_spec"]
logging_orders = [re.findall(r'\(\s*"([a-z_]+)"', b) for b in seg_orders]
logging_orders = [o for o in logging_orders if "radial" in o]
check(
    "logging _pack_segments order unchanged",
    all(o == expected for o in logging_orders),
    str(logging_orders),
)

print("\n== buffer arity derives from len(RADIAL_METRIC_NAMES) ==")
check(
    "no hard-coded 15 for radial width",
    not re.search(r"radial\w*\s*[*]\s*15\b|15\s*[*]\s*radial", src),
)
check(
    "num_radial_types from len()",
    src.count("len(RADIAL_METRIC_NAMES)") >= 3,
    f"count={src.count('len(RADIAL_METRIC_NAMES)')}",
)

print("\n== spectrum pops tolerate absence ==")
check('no bare pop("spectrum")', '.pop("spectrum")' not in src)

print("\n== per-rank logging: right store-gate per family ==")
# The fsdp/ddp/expert families own disjoint parameter slices, so under
# per-rank logging every rank must keep its own metrics -> _stores_norms.
# scalar/embed are REPLICATED (identical on every rank), so keeping them
# everywhere would log world_size duplicates of one series -> pinned to
# local rank 0 via _stores_replicated_norms. Mixing the two up is silent:
# the wrong gate either drops a shard's metrics or duplicates a global one.
_expect_gate = {
    "step_scalar": "_stores_replicated_norms",
    # step_embedding round-robins its params across shard ranks via
    # `_owns_replicated_param` and `continue`s on non-owners, so by the time
    # the store gate is reached `final_norms` already holds only this rank's
    # share -- hence the plain `_stores_norms` here, unlike step_scalar.
    "step_embedding": "_stores_norms",
    "step_experts": "_stores_norms",
    "step_ddp": "_stores_norms",
    "_gather_and_log_fsdp": "_stores_norms",
}
_tree = ast.parse(src)
_fns = {n.name: n for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef)}
for _fn, _want in _expect_gate.items():
    _node = _fns.get(_fn)
    _gates = (
        sorted(
            {
                a.attr
                for a in ast.walk(_node)
                if isinstance(a, ast.Attribute)
                and a.attr in ("_stores_norms", "_stores_replicated_norms")
            }
        )
        if _node is not None
        else []
    )
    check(f"{_fn}: gate is {_want}", _gates == [_want], f"found {_gates}")

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL DISCO ORDERING CHECKS PASSED")
