# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for `sequence_wise_aux_loss` document segmentation.

The reference is written FROM DeepSeek-V3 Eq. 17-20 and the token-weighting
rule, with a Python document loop in float64. It deliberately shares no code
with the implementation: a reference copied from the code under test agrees
with the code's bugs to 0 ULP.

Run: python tests/unit_tests/opt_moe/test_aux_loss_segmentation.py
"""

import torch

from torchtitan.models.opt_moe.norm_moe import sequence_wise_aux_loss

FAILURES = []
SKIPPED = []


def check(name, cond, detail=""):
    print(f"  {name:44} {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def reference(scores, indices, top_k, alpha, loss_mask=None, positions=None, segs=None):
    """DeepSeek-V3 Eq. 17-20, token-weighted mean over documents, float64.

    Eq 19: s'_{i,t} = s_{i,t} / sum_j s_{j,t}
    Eq 20: P_i = (1/T_d) sum_t s'_{i,t}
    Eq 18: f_i = (N / (K * T_d)) * #{t in d : i in TopK(t)}
    Eq 17: L_d = alpha * sum_i f_i * P_i
    Total: sum_d T_d * L_d / sum_d T_d
    """
    sc = scores.double()
    T, N = sc.shape
    mask = (
        torch.ones(T, dtype=torch.bool)
        if loss_mask is None
        else loss_mask.reshape(-1).bool().cpu()
    )
    if segs is not None:
        # Segmentation supplied by the caller from the document lengths it used
        # to BUILD `positions`. This is the only form that can catch the
        # segmentation RULE being wrong -- deriving it from `positions` with
        # `pos[t] != pos[t-1] + 1` re-states the implementation's own premise.
        segs = [list(r) for r in segs]
    elif positions is None:
        segs = [list(range(T))]
    else:
        pos = positions.cpu().tolist()
        segs, cur = [], []
        for t in range(T):
            # A segment starts at a document reset OR a CP shard seam, i.e.
            # wherever the position is not the previous position plus one.
            if cur and pos[t] != pos[t - 1] + 1:
                segs.append(cur)
                cur = []
            cur.append(t)
        if cur:
            segs.append(cur)

    losses, weights = [], []
    for seg in segs:
        toks = [t for t in seg if mask[t]]
        if not toks:
            continue
        t_d = len(toks)
        sp = sc[toks] / sc[toks].sum(dim=-1, keepdim=True).clamp_min(1e-300)
        p_i = sp.mean(dim=0)
        cnt = torch.zeros(N, dtype=torch.float64)
        for t in toks:
            for e in indices[t].tolist():
                cnt[e] += 1
        f_i = cnt * N / (top_k * t_d)
        losses.append(alpha * (f_i * p_i).sum())
        weights.append(float(t_d))
    if not losses:
        return torch.tensor(0.0, dtype=torch.float64)
    loss = torch.stack(losses)
    w = torch.tensor(weights, dtype=torch.float64)
    return (loss * w).sum() / w.sum()


def _unweighted_reference(scores, indices, top_k, alpha, loss_mask, positions):
    """Pre-change behaviour: unweighted mean over non-empty documents.

    Only used to show that token weighting actually changes the result on
    ragged input; the correctness reference is `reference` above.
    """
    sc = scores.double()
    t_len, n_exp = sc.shape
    mask = (
        torch.ones(t_len, dtype=torch.bool)
        if loss_mask is None
        else loss_mask.reshape(-1).bool().cpu()
    )
    pos = positions.cpu().tolist()
    segs, cur = [], []
    for t in range(t_len):
        if cur and pos[t] != pos[t - 1] + 1:
            segs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        segs.append(cur)
    vals = []
    for seg in segs:
        toks = [t for t in seg if mask[t]]
        if not toks:
            continue
        t_d = len(toks)
        sp = sc[toks] / sc[toks].sum(dim=-1, keepdim=True).clamp_min(1e-300)
        p_i = sp.mean(dim=0)
        cnt = torch.zeros(n_exp, dtype=torch.float64)
        for t in toks:
            for e in indices[t].tolist():
                cnt[e] += 1
        vals.append(alpha * ((cnt * n_exp / (top_k * t_d)) * p_i).sum())
    return float(torch.stack(vals).mean()) if vals else 0.0


def case(name, t_len, n_exp, top_k, positions, loss_mask=None, seed=0, tol=2e-6):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(t_len, n_exp, generator=g) + 0.01
    indices = torch.stack(
        [torch.randperm(n_exp, generator=g)[:top_k] for _ in range(t_len)]
    )
    got = sequence_wise_aux_loss(
        scores,
        indices,
        B=1,
        S=t_len,
        top_k=top_k,
        loss_mask=loss_mask,
        positions=positions,
    )
    exp = reference(scores, indices, top_k, 1.0, loss_mask, positions)
    rel = abs(got.double().item() - exp.item()) / max(abs(exp.item()), 1e-30)
    check(name, rel < tol, f"rel={rel:.2e}")


def main():
    print("segmentation vs equation-derived reference")
    case(
        "3 documents, clean start",
        12,
        8,
        2,
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4]),
    )
    case(
        "leading partial document (CP shard)",
        12,
        8,
        2,
        torch.tensor([5, 6, 7, 0, 1, 2, 0, 1, 2, 3, 4, 5]),
    )
    case("all length-1 documents", 12, 8, 2, torch.zeros(12, dtype=torch.long))
    case("single document", 12, 8, 2, torch.arange(12))
    case("T=1", 1, 8, 2, torch.tensor([0]))
    # A CP "headtail" shard is two DISJOINT chunks concatenated; the join
    # carries no positions==0, so a reset-only rule merges two documents.
    case(
        "headtail CP seam",
        16,
        8,
        2,
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27]),
    )
    case(
        "padded tail with loss_mask",
        12,
        8,
        2,
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 0, 0, 0, 0]),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool),
    )
    g = torch.Generator().manual_seed(3)
    lens = torch.randint(1, 40, (30,), generator=g).tolist()
    case(
        "512 tokens, 30 ragged documents",
        512,
        16,
        4,
        torch.cat([torch.arange(L) for L in lens])[:512],
    )

    print("independent segmentation (does not reuse the implementation's rule)")
    for lengths in ([7, 1, 12, 3], [1, 1, 1, 1, 20], [30]):
        t_len = sum(lengths)
        g2 = torch.Generator().manual_seed(17)
        sc2 = torch.rand(t_len, 8, generator=g2) + 0.01
        ix2 = sc2.topk(2, dim=-1).indices
        pos2 = torch.cat([torch.arange(L) for L in lengths])
        # TRUE segments, from the lengths -- never derived from `positions`.
        bounds, off = [], 0
        for L in lengths:
            bounds.append(range(off, off + L))
            off += L
        got = sequence_wise_aux_loss(
            sc2, ix2, B=1, S=t_len, top_k=2, positions=pos2
        )
        exp = reference(sc2, ix2, 2, 1.0, None, None, segs=bounds)
        rel = abs(got.double().item() - exp.item()) / abs(exp.item())
        check(
            f"lengths={lengths}: matches independently-segmented reference",
            rel < 2e-6,
            f"rel={rel:.2e}",
        )

    print("token-weighting properties")
    # Use the router's REAL top-k, not random indices. Random indices are
    # statistically independent of `scores`, which zeroes the f/P correlation
    # these properties are about and makes the checks vacuous.
    n_exp, top_k = 64, 4
    g = torch.Generator().manual_seed(5)
    t_len = 2048
    scores = torch.rand(t_len, n_exp, generator=g) + 0.01
    indices = scores.topk(top_k, dim=-1).indices
    kw = dict(B=1, S=t_len, top_k=top_k)

    one_doc = sequence_wise_aux_loss(
        scores, indices, positions=torch.arange(t_len), **kw
    ).item()
    whole = sequence_wise_aux_loss(scores, indices, **kw).item()
    check(
        "single doc == whole-stream fallback",
        abs(one_doc - whole) / whole < 1e-5,
        f"{one_doc:.9f} vs {whole:.9f}",
    )

    # Weighting only differs from an unweighted mean on RAGGED documents -- for
    # equal-length documents the two are bit-identical, so an equal-length
    # check cannot test weighting at all. Build a ragged stream: many 1-token
    # documents alongside a few long ones, which is the `best_fit` pad shape.
    long_len, n_long, n_short = 500, 3, 548
    pos = torch.cat(
        [torch.arange(long_len) for _ in range(n_long)]
        + [torch.zeros(n_short, dtype=torch.long)]
    )
    assert pos.numel() == t_len
    weighted = sequence_wise_aux_loss(scores, indices, positions=pos, **kw).item()
    unweighted = _unweighted_reference(scores, indices, top_k, 1.0, None, pos)
    long_only = sequence_wise_aux_loss(
        scores[: long_len * n_long],
        indices[: long_len * n_long],
        B=1,
        S=long_len * n_long,
        top_k=top_k,
        positions=pos[: long_len * n_long],
    ).item()
    check(
        "ragged: weighting changes the result",
        abs(weighted - unweighted) > 0.1,
        f"weighted={weighted:.5f} unweighted={unweighted:.5f}",
    )
    check(
        "ragged: weighting moves toward the long documents",
        abs(weighted - long_only) < abs(unweighted - long_only),
        f"|w-long|={abs(weighted-long_only):.5f} |u-long|={abs(unweighted-long_only):.5f}",
    )

    # A loss_mask is what removes padding EXACTLY.
    mask = torch.zeros(t_len, dtype=torch.bool)
    mask[: long_len * n_long] = True
    masked = sequence_wise_aux_loss(
        scores, indices, positions=pos, loss_mask=mask, **kw
    ).item()
    check(
        "masked padding == real tokens only",
        abs(masked - long_only) / long_only < 1e-5,
        f"{masked:.9f} vs {long_only:.9f}",
    )

    # The f_i side must be masked too. Dropping `* valid` from `per_tok` is
    # invisible on fully-masked documents and ~39% wrong on partially-masked
    # ones -- which is exactly the SFT shape (prompt tokens masked mid-document).
    part = torch.ones(t_len, dtype=torch.bool)
    part[100:400] = False  # mask a slice INSIDE the first document
    got = sequence_wise_aux_loss(scores, indices, positions=pos, loss_mask=part, **kw)
    exp = reference(scores, indices, top_k, 1.0, part, pos)
    rel = abs(got.double().item() - exp.item()) / abs(exp.item())
    check("partially-masked document matches reference", rel < 2e-6, f"rel={rel:.2e}")

    print("CP headtail seam (doc_id)")
    # "headtail" gives rank r the global chunks r and 2*cp-1-r concatenated, so
    # a rank's tokens are two NON-adjacent slices of the global stream.
    #
    # GROUND TRUTH, stated once and derived from the global indices rather than
    # from the rule under test: a new segment starts at a token whose global
    # index is not its predecessor's + 1, OR whose document differs. The
    # implementation approximates the first half with `positions != prev + 1`
    # and the second with `doc_id`, and BOTH halves are load-bearing:
    #   - many short documents  -> doc_id catches a seam between two documents
    #                              that positions alone reads as a continuation
    #   - ONE long document     -> doc_id never changes, and only the positions
    #                              term catches the seam
    # so each case below must fail if either half is removed.
    n_exp, top_k = 16, 4
    # The chunk boundary must land MID-document, or the seam coincides with a
    # real document start and `positions` alone already handles it (measured
    # 0.0000% error whenever chunk is a multiple of the document length).
    # cp=2, 6 documents of 64 -> chunk 96, i.e. 32 tokens into a document.
    for label, seq, n_seq, cp_deg in (
        ("many short docs", 64, 6, 2),
        ("ONE long doc", 512, 1, 4),
    ):
        t_len = seq * n_seq
        g = torch.Generator().manual_seed(3)
        pos_g = torch.cat([torch.arange(seq) for _ in range(n_seq)])
        doc_g = torch.cat(
            [torch.full((seq,), d, dtype=torch.int32) for d in range(n_seq)]
        )
        sc_g = torch.rand(t_len, n_exp, generator=g) + 0.01
        ix_g = sc_g.topk(top_k, dim=-1).indices
        chunk = t_len // (2 * cp_deg)
        any_seam_matters = False
        for rank in range(cp_deg):
            gidx = torch.cat(
                [
                    torch.arange(rank * chunk, (rank + 1) * chunk),
                    torch.arange(t_len - (rank + 1) * chunk, t_len - rank * chunk),
                ]
            )
            pos, did, sc, ix = pos_g[gidx], doc_g[gidx], sc_g[gidx], ix_g[gidx]
            # ground truth segmentation, from global indices + document identity
            bounds, start = [], 0
            for t in range(1, len(gidx)):
                if gidx[t] != gidx[t - 1] + 1 or did[t] != did[t - 1]:
                    bounds.append((start, t))
                    start = t
            bounds.append((start, len(gidx)))
            pos_true = torch.cat([torch.arange(b - a) for a, b in bounds])
            exp = reference(sc, ix, top_k, 1.0, None, pos_true).item()
            kwr = dict(B=1, S=len(gidx), top_k=top_k, positions=pos)
            both = sequence_wise_aux_loss(sc, ix, doc_id=did, **kwr).item()
            check(
                f"{label} rank {rank}: matches ground truth",
                abs(both - exp) / exp < 2e-6,
                f"rel={abs(both-exp)/exp:.2e}",
            )
            # positions-only: what the code did before doc_id existed
            pos_only = sequence_wise_aux_loss(sc, ix, **kwr).item()
            # doc_id-only: drop the positions term from the rule
            seg = torch.ones_like(pos, dtype=torch.bool)
            seg[1:] = did[1:] != did[:-1]
            synth = torch.cat(
                [
                    torch.arange(int(n))
                    for n in torch.diff(
                        torch.cat([seg.nonzero().flatten(), torch.tensor([len(gidx)])])
                    )
                ]
            )
            id_only = sequence_wise_aux_loss(
                sc,
                ix,
                B=1,
                S=len(gidx),
                top_k=top_k,
                positions=synth,
            ).item()
            if abs(pos_only - exp) / exp > 1e-4 or abs(id_only - exp) / exp > 1e-4:
                any_seam_matters = True
        check(
            f"{label}: a half-rule IS wrong here (both halves load-bearing)",
            any_seam_matters,
        )

    print("numerics")
    g = torch.Generator().manual_seed(0)
    t_len, n_exp, top_k = 24, 8, 2
    s64 = (
        torch.rand(t_len, n_exp, generator=g, dtype=torch.float64) + 0.01
    ).requires_grad_(True)
    idx = torch.stack(
        [torch.randperm(n_exp, generator=g)[:top_k] for _ in range(t_len)]
    )
    pos = torch.cat([torch.arange(8) for _ in range(3)])
    fn64 = lambda x: sequence_wise_aux_loss(  # noqa: E731
        x, idx, B=1, S=t_len, top_k=top_k, positions=pos
    )
    check(
        "gradcheck float64",
        torch.autograd.gradcheck(fn64, (s64,), eps=1e-6, atol=1e-8, rtol=1e-5),
    )
    check("fp64 not downcast", fn64(s64).dtype == torch.float64)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        x = (torch.rand(t_len, n_exp, generator=g) + 0.01).to(dt)
        out = sequence_wise_aux_loss(
            x, idx, B=1, S=t_len, top_k=top_k, positions=pos
        )
        check(f"output dtype tracks {dt}", out.dtype == dt)

    all_masked = sequence_wise_aux_loss(
        s64.detach().float().requires_grad_(True),
        idx,
        B=1,
        S=t_len,
        top_k=top_k,
        positions=pos,
        loss_mask=torch.zeros(t_len, dtype=torch.bool),
    )
    check("all tokens masked -> 0", all_masked.item() == 0.0)

    if not torch.cuda.is_available():
        # These two are CUDA-only. Record them so the summary cannot claim a
        # clean run: reverting `acc_dtype` to `scores.dtype` is invisible on
        # CPU (bf16 scatter_add is exact there) and the suite would go green
        # with the bug present.
        SKIPPED.append("bf16 accumulation (CUDA-only)")
        SKIPPED.append("fullgraph compile bit-identical (CUDA-only)")
    if torch.cuda.is_available():
        # The fp32 accumulation is load-bearing and CUDA-only: bf16
        # `scatter_add_` of 2048 ones returns 256.0 on CUDA (2048.0 on CPU), so
        # accumulating counts in `scores.dtype` makes a single long document
        # come out ~4x wrong. A dtype-only assertion does not catch it -- this
        # needs a VALUE check, on CUDA, with a document longer than 256 tokens.
        print("bf16 accumulation (CUDA)")
        t_bf, n_bf, k_bf = 2048, 32, 4
        gb = torch.Generator().manual_seed(9)
        sc_bf = (torch.rand(t_bf, n_bf, generator=gb) + 0.01).cuda()
        ix_bf = sc_bf.topk(k_bf, dim=-1).indices
        pos_bf = torch.arange(t_bf, device="cuda")  # ONE long document
        kwb = dict(B=1, S=t_bf, top_k=k_bf, positions=pos_bf)
        ref_bf = sequence_wise_aux_loss(sc_bf.double(), ix_bf, **kwb).item()
        got_bf = sequence_wise_aux_loss(sc_bf.bfloat16(), ix_bf, **kwb).double().item()
        check(
            "bf16 scores: counts do not saturate",
            abs(got_bf - ref_bf) / ref_bf < 0.02,
            f"bf16={got_bf:.6f} fp64={ref_bf:.6f}",
        )

        print("compile")
        dev = "cuda"
        t_len, n_exp, top_k = 256, 16, 4
        sc = (torch.rand(t_len, n_exp, generator=g) + 0.01).to(dev)
        ix = torch.stack(
            [torch.randperm(n_exp, generator=g)[:top_k] for _ in range(t_len)]
        ).to(dev)
        pos = torch.cat([torch.arange(32) for _ in range(8)]).to(dev)
        kw = dict(B=1, S=t_len, top_k=top_k, positions=pos)
        a = sequence_wise_aux_loss(sc, ix, **kw)
        b = torch.compile(sequence_wise_aux_loss, fullgraph=True)(sc, ix, **kw)
        check("fullgraph compile bit-identical", torch.equal(a, b))

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        raise SystemExit(1)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): {SKIPPED}")
        print("all RUN checks passed -- NOT a clean bill of health, see SKIPPED above")
    else:
        print("all checks passed")


if __name__ == "__main__":
    main()
