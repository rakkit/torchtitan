# DiSCO optimizer

`DiSCO` (`disco.py`) is a Muon/orthogonalized-update-style optimizer: instead of applying the raw
gradient, it computes an LMO ("linear minimization oracle") update via `AbstractDiSCO.lmo`
(`abstract_disco.py`) — typically a Newton-Schulz zeropower iteration that orthogonalizes the
gradient matrix (or a cheaper per-row normalization for embeddings, see below) — then applies
`w = w*(1 - wd*lr) - lr*u`.

This file documents **how DiSCO handles different parameter types under different parallelism
strategies**, which is most of what's structurally interesting about `disco.py`. Norm/gram/spectrum
tracking (a secondary concern layered on top) is covered at the end.

## Files

- `disco.py` — the `DiSCO` optimizer itself; all parallelism-specific logic lives here.
- `abstract_disco.py` — `AbstractDiSCO` base class: `lmo()`, `normalise_grad()` (the actual
  update-shaping math per `norm_factor`), and norm/gram tracking *state*
  (`need_to_calculate_norm`, `norms_to_log`, `gram_level`/`gram_scalar_names`/`gram_vector_names`,
  `norms_at_current_step`).
- `norm_helper.py` / `gram_helper.py` — see "Norm/gram/spectrum tracking" below.
- `pre_norm_helper.py` — see "Pre-norm: a stage before LMO" below.
- `spectrum_logging.py` — turns raw `track_spectrum_*` tensors into W&B images / Parquet export.
- `gram_vector_logging.py` — turns raw `track_gram_*` vector tensors into W&B atlas-grid images
  (index vs. value line plots, one grid per gram metric name — copy-and-adapted from
  `spectrum_logging.py`'s grid/layout machinery, kept as its own module so `spectrum_logging.py`
  stays untouched) and/or a Parquet export (mirrors `spectrum_logging._export_spectrum`) — see
  "Norm/gram/spectrum tracking" below.
- `readme.md` — this file.

## Why parameter type matters here (not just "which mesh shards it")

A parameter is routed to one of 5 handlers, and **the routing is driven by the LMO algorithm the
parameter needs, not simply by which mesh(es) shard it**. This is the thing most worth
internalizing before touching this file:

| Handles | Routed by | Algorithm | Needs full (unsharded) matrix for the *update itself*? |
|---|---|---|---|
| `step_scalar` | `p.numel() == 1` | `sign(grad)` | n/a (scalar) |
| `step_embedding` | `backend == "identity"` and `norm_factor` starts with `embed`/`unembed` | per-**row** L2 normalization (`fused_embed_linear` etc., `abstract_disco.py`) | **No** — row-separable |
| `step_experts` | structural: `ndim == 3` (MoE routed experts) | Newton-Schulz orthogonalization, per expert | No extra gather — EP shards along the *expert* axis, so each rank already holds each of its owned experts' full 2-D matrix |
| `step_ddp` | structural: not FSDP/EP-sharded | Newton-Schulz orthogonalization | Cheap — `dp_replicate` means every rank already has a full replica (at most a TP-only gather) |
| `step_fsdp` | structural: FSDP-sharded | Newton-Schulz orthogonalization | **Yes, expensive** — FSDP shards along the matrix's own row dimension, so the whole matrix must be reconstructed via `all_to_all_single` before the algorithm can run |

Routing is decided in `_build_param_lists` (called once at optimizer construction, cached
thereafter): scalars first, then `_is_embed_group(group)` (checks `backend`/`norm_factor` on the
param's *group*, i.e. a config/algorithm choice — **completely independent of how the parameter is
actually sharded**), then `get_param_type(p, fsdp_enabled, expert_enabled)` (a structural check —
`ndim == 3` → Expert, else FSDP or DDP depending on `fsdp_enabled`) for everything else.

### The key correction: `step_embedding` params *can* be FSDP-sharded

An earlier version of this doc claimed `step_embedding` params never carry an FSDP shard. That's
wrong. Embedding/unembedding parameters are typically large (vocab-sized) and commonly *are*
FSDP-sharded (row-sharded, same as any other FSDP param) in real configs — `step_embedding` never
routes through the structural `get_param_type` check at all (it short-circuits on the group's
`backend`/`norm_factor` before that check ever runs), so an FSDP-sharded embedding weight still
ends up in `step_embedding`, not `step_fsdp`.

The reason `step_embedding` doesn't need `step_fsdp`'s expensive bucketed `all_to_all_single`
reconstruction is **not** "no sharding survives to this path" — it's that the embedding/unembedding
LMO (`fused_embed_linear`/`fused_embed_sqrt`/`fused_unembed_linear`/`fused_unembed_sqrt` in
`abstract_disco.py`) normalizes **per row** (`row_l2_norm = g.pow(2).sum(dim=-1, ...).sqrt()`,
purely local to each row) — unlike the Newton-Schulz orthogonalization `step_ddp`/`step_fsdp`/
`step_experts` use, which is not row-separable and genuinely needs the whole matrix at once. Since
FSDP shards along dim 0 (rows), each rank's local shard already contains everything the embedding
LMO needs for *its own rows* — zero communication required for the update itself. `step_embedding`
still calls `p.full_tensor()` / `get_momentum_or_grad(..., gather_to_local=True)`, but only inside
the *norm-logging* loop (SVD-based norms genuinely do need the whole matrix), never for the update
path. That's a real, separate collective per parameter, not batched the way `step_fsdp` batches
multiple params into one bucket-wide `all_to_all_single` — a plausible future optimization if the
embed param count/sharding ever makes it matter, not something built today.

## How `disco.py` is put together (read this before changing anything in it)

Reverse-engineered while working on the logging path; collected here because every one of these
had to be rediscovered from the code, and several of them are load-bearing invariants that are
easy to break silently.

### Which `step_*` a parameter goes to

Decided once in `_build_param_lists`, not per step:

1. `p.numel() == 1` -> `scale_params` -> `step_scalar` (asserts identity backend / sign norm
   factor / identity pre_norm). These never reach `calculate_norm` at all.
2. the param's group is an "embed group" (`backend == "identity"` and `norm_factor` starts with
   `embed`/`unembed`) -> `embed_params` -> `step_embedding`. This includes `output`/lm_head.
3. otherwise `get_param_type(p, fsdp_enabled, expert_enabled)` -> DDP / FSDP / Expert.
   **A 3-D parameter becomes an Expert param when `expert_enabled OR fsdp_enabled`** - so plain
   FSDP with `expert_parallel_degree=1` still exercises `step_experts`. EP is not required to
   test that path.

Note a 1-D parameter of length > 1 (an affine norm weight) is NOT a scale param; it goes to
DDP/FSDP and reaches `calculate_norm`, which is why the 1-D case there matters.

### Ownership - a different scheme per family

Nothing here is shared; each family answers "who computes this parameter's metrics" its own way:

- **FSDP**: `param_idx % world_size == rank` (`_fsdp_rank_owned_param_indices`).
- **DDP**: `_ddp_owner_rank_by_param` / `_ddp_owner_bucket_by_param`, built by sorting params
  big -> small to reduce padding.
- **Experts**: every rank holds `ep_per_rank = ceil(global_experts / fsdp_world_size)` experts of
  *every* expert param, and the global index is `actual_ep_idx = e + r * ep_per_rank`.

The union across ranks is exactly the full set in all three cases, which is what makes per-rank
logging lossless.

### Expert blocking, and why the LMO is only two calls

`_expert_blocks` groups expert params into shape-homogeneous blocks - typically `w1`/`w3` share a
shape and `w2` another, giving blocks `(0, 2L)` and `(2L, 3L)` for `L` MoE layers. Each block's
gradients are packed into one `big_g` of `[K * ep_per_rank, A, B]` and the LMO is called **once
per block**. So for qwen30b-a3b at EP=64 the whole expert update is two batched calls of
`[188, 768, 2048]` and `[94, 2048, 768]` per step - not hundreds of small ones. Any cost model
that assumes per-expert LMO calls is wrong.

### `step_fsdp` runs three all-to-alls, and one is logging-only

1. gradients out, so the owner rank gets `full_g` and can run `self.lmo(full_g)`;
2. the update back, so every rank gets its shard of every parameter's update;
3. **the weights**, purely to rebuild `full_weight` for the metrics.

(3) exists because after (1) and (2) a rank holds the full *gradient* and *update* for the
parameters it owns, but still only its own row-shard of the *weight*. It is gated on
`need_to_calculate_norm` and is by far the largest logging-only traffic. It cannot be removed
while `spectrum` / `condition_number` / `effective_rank` are wanted, because those need the
assembled matrix; everything else (Frobenius-type quantities, and all of radial) is a sum over
entries and would reconstruct exactly from small all-reduces.

### Metrics are packed into one flat buffer per rank

`_pack_segments` concatenates present segments in a fixed order --
`upd, w, gram, gram_vec, radial, upd_spec, w_spec` -- and returns the offsets it actually used, so
they cannot drift from pack order. Absent segments are simply skipped, and every reader is keyed
on presence (`if "upd_spec" in offsets`), which is what lets the spectrum be dropped entirely.

`step_experts` is the exception: it builds its buffer by hand and hand-computes offsets
(`weight_scalar_offset = expected_total`, `gram_scalar_offset = 2 * expected_total`, ...). That
only stays correct because `expected_total_gram == 0` when gram is off. Reordering or adding a
segment there means editing two places - the drift class `_pack_segments` was written to prevent.

### Invariants that are easy to break silently

- **Fixed arity.** Flat buffer sizes come from `len(RADIAL_METRIC_NAMES)`, `len(norms_to_log)` and
  `gram_scalar_names(level)`. Every metric function must return the *same key set for every
  input* - degenerate, 1-D, 3-D, missing optional inputs - filling with a 0 sentinel rather than
  omitting keys. `calculate_gram_metrics` returning `{}` for `level <= 0` is fine only because
  `G == 0` is then consistent everywhere.
- **Ordering.** norm/gram/radial all run *before* the real parameter apply, so `p` /
  `full_weight` are genuinely pre-update. `pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)`
  stands in for the post-update weight. It is purely elementwise
  (`w*(1 - wd*lr) - lr*u`), so it decomposes across shards - `pseudo_w[shard]` equals
  `pseudo(W[shard], u[shard])`.
- **`U` is not `-lr*u` when `wd != 0`.** radial's displacement is
  `pseudo_w - W_before = -lr*u - wd*lr*W_before`. Reusing the update's spectrum as
  `sigma_update` is only valid at `wd == 0`.
- **`need_to_calculate_norm` lifecycle.** Set by `calculate_norm_at_next_step` (which also clears
  `norms_at_current_step` and refreshes the gram names) and cleared at the end of `step()`. It is
  the single gate for the whole logging path.
- **Two gates, not one, for "does this rank keep metrics".** The unpack gate decides who reads
  the buffer; `_stores_norms` decides who keeps the result. Moving one without the other silently
  discards work.

## Per-path structure (high level — see the docstrings/comments in each `step_*` for exact mechanics)

- **`step_scalar`**: trivial — `sign(grad)`, `@torch.compile()`-decorated, no distributed
  reconstruction of anything.
- **`step_embedding`**: gradients fetched per-param (`gather_to_local=False` for the actual
  update — local shard is enough), LMO applied locally, update applied via
  `_update_embed_params_fast` (batched by shape where possible for `_foreach_*` fusion). Norm/gram
  logging (when `need_to_calculate_norm`) additionally gathers full tensors per param — see below.
- **`step_experts`**: MoE routed-expert weights, shape `(num_experts_per_block, D, D')`, grouped
  into same-shape "blocks" (`_expert_blocks`, `_precompute_experts_metadata`) so all experts in a
  block LMO together as one batched call. Expert-Parallel shards along the *expert* axis over the
  FSDP mesh (`ep_per_rank = ceil(num_experts / fsdp_mesh.size())`) — each rank's owned experts are
  already complete 2-D matrices, no reconstruction collective needed.
- **`step_ddp`**: DDP-replicated params (`ndim <= 2` only — MoE/3-D params are forced through
  `step_experts`/`step_fsdp` instead, see the `invalid_ddp_params` check in
  `_precompute_ddp_metadata`). Each rank computes LMO for its own "owned" subset
  (`_ddp_owned_indices`, a round-robin partition purely for *avoiding duplicate work/logging*
  across replicas — not sharding, since `dp_replicate` means full replicas), then one flat
  `all_gather` distributes everyone's computed updates to everyone (Phase B) before the batched
  apply (Phase C).
- **`step_fsdp`**: the expensive path. Params are grouped into buckets of `world_size` params each
  (`_fsdp_bucket_ranges`); each bucket does a forward `all_to_all_single` to reconstruct the full
  gradient for whichever param this rank owns in that bucket, runs LMO, then a reverse
  `all_to_all_single` to re-scatter the update back to shards for the apply. Two implementations
  exist, chosen by `DISCO_FSDP_A2A_MODE` env var (default `"once"`):
  - `"once"` (`use_global_fast_path`): all buckets' forward/reverse communication is batched into
    **one** pair of global `all_to_all_single` calls (`_prepare_fsdp_lmo` fills one big send
    buffer for everything up front) — the fast path, and the only one exercised by default.
  - `"bucket"`: one `all_to_all_single` pair *per bucket* — a fallback, apparently rarely
    exercised (see the bug below, which only this path hit).

## Pre-norm: a stage before LMO

`AbstractDiSCO.lmo` fuses "orthogonalize" (Newton-Schulz, needs the full matrix) with "post-norm"
(`normalise_grad`). **Pre-norm** is a third, earlier stage applied to the effective gradient (raw
grad, or momentum-blended buffer if `momentum > 0`) **before any communication for LMO** — so it
runs on the raw tensor in its original dtype, before the communication-dtype downcast. Configured
per group via `pre_norm` (defaults to `"identity"`, a no-op), same override mechanism
(`extra_param_group_split_rules`) as `norm_factor`/`backend`.

Only matters for `step_fsdp`/`step_embedding` — `step_ddp`/`step_experts` already have the full
matrix locally (DDP replication; EP shards along the expert axis only), so pre-norm there is a
direct computation, no special handling.

Config values look like `"row-l2"`/`"col-l2"`/`"mat-l2"` — the prefix before the first `-`
(`row`/`col`/`mat`) selects the communication strategy, the full string selects the formula
(`pre_norm_helper.py`'s `PRE_NORM_*` registries), so later variants (`"col-rms"`, ...) slot in
without touching dispatch code:

- **row** (`row-l2`): reduces along dim=-1, which is never the FSDP-sharded dimension — a local
  shard already holds complete rows, so this is **zero communication**. Applied inline, directly to
  `g` (DTensor or plain Tensor, whichever it already is) at the exact point it's fetched in
  `get_momentum_or_grad`/`get_momentum_or_grad_list`/`_get_effective_grad_by_group`
  (`_apply_row_pre_norm`) — no separate pass, no cache, no `.to_local()` unwrap. dim=-1 reduction on
  a DTensor dispatches locally per shard, same as any other local op.
- **col**/**mat** (`col-l2`/`mat-l2`): reduce along the FSDP-sharded dimension / the whole matrix —
  genuinely need combining across ranks. `_apply_reduce_pre_norm_pass` (called once per step, right
  after `prepare_gradients_and_momentum`) does this in two batched phases instead of a per-param
  loop:
  1. Groups every col/mat param by `(is_fsdp_row_sharded, exact pre_norm string, local_shard_shape,
     eps)` (`_precompute_pre_norm_metadata`, once at init — mirrors the existing
     `_embed_extra_shape_groups`/FSDP-bucket shape-grouping pattern), then computes **one**
     `torch.stack` + one vectorized reduction per shape group instead of N individual per-param
     calls.
  2. Every FSDP-sharded group's partial sum-of-squares gets packed into **one** buffer
     (`_pack_segments`, same helper `step_ddp`/`step_fsdp` already use for norm-logging fusion) for
     a **single** `dist.all_reduce`, regardless of how many groups or params exist that step.
     Non-sharded groups (DDP/experts/non-FSDP-sharded embed) skip the all-reduce entirely — their
     local view is already the full tensor.

  Results are cached in `self._pre_normed_grad_cache` (keyed by `id(p)`); the 3 effective-grad
  fetchers check this cache first and return directly instead of recomputing. `gather_to_local=True`
  callers (only `step_embedding`'s norm-logging re-fetch, gated behind `need_to_calculate_norm`, not
  the hot path) bypass the cache and reapply the full-tensor formula fresh once already gathered —
  mathematically identical, no all-reduce needed once the data is whole.

**Known limitation — TP composition is out of scope.** A param that's *both* FSDP/DDP-sharded and
TP-sharded isn't handled correctly: row-norm assumes dim=-1 isn't TP-sharded (breaks under
TP col-parallel, `Shard(dim=1)`); col/mat's "already full" branch for DDP/experts/non-sharded embed
does a plain `.to_local()`, not `_prepare_ddp_lmo`'s TP-aware gather. Pre-norm is approximate under
TP composition, not solved in this pass — same scoping decision made explicitly up front, not
discovered as a gap.

**Known limitation — 1-D params aren't supported.** Row/col/mat all assume a matrix shape
`[rows, cols]` where FSDP shards dim 0 and dim=-1 is a separate, unsharded axis. For a genuinely
1-D param (e.g. a bias vector — a real, supported case elsewhere, see `lmo()`'s `ndim==1` branch),
dim=-1 *is* dim 0 *is* the sharded dim, so row's "dim=-1 is never sharded" premise and col/mat's
row-vs-column distinction both collapse — worse, silently stacking several different 1-D params
together would reduce *across params* instead of within one. `_precompute_pre_norm_metadata` raises
a clear `ValueError` at init if a non-`identity` `pre_norm` is configured on a `<2`-D param (and
scalar/`step_scalar` groups assert `pre_norm == "identity"` outright) rather than computing
something silently wrong — use `"identity"` for bias-like params for now.

## A real, pre-existing bug found while reviewing this file

`step_fsdp`'s bucketed (`"bucket"` mode) branch read `device=workspace["device"]` when allocating
`bucket_workspace`. **`"device"` was never actually a key in the `workspace` dict, in either
mode** — `_create_fsdp_step_workspace`/`_allocate_fsdp_once_workspace` never set it, and `step()`'s
dispatch only even *builds* a `workspace` dict when `fsdp_a2a_mode == "once"` (it stays `None`
otherwise). So setting `DISCO_FSDP_A2A_MODE=bucket` with any FSDP params crashed on the very first
optimizer step (`TypeError: 'NoneType' object is not subscriptable`) — **the bucketed fallback mode
was completely non-functional**. Fixed by using `step_fsdp`'s own local `device` variable (already
computed at the top of the function from `fsdp_params[0].device`) instead of threading it through
`workspace`. This bug predates all other work in this doc (confirmed via `git show HEAD`) — it
wasn't introduced by the norm/gram work below, just found while reading through the whole file.
If you ever need `"bucket"` mode for real, re-verify it end-to-end — this path looks
under-exercised (this bug would have been immediately obvious the first time it actually ran).

## Norm/gram/spectrum tracking

Every path also optionally logs, per parameter, gated by a single flag
`self.need_to_calculate_norm` (set externally via `calculate_norm_at_next_step()`, driven by
`config.metrics.log_norm_freq`):

- `track_update_*` — scalar norms (`norm_helper.calculate_norm`) of the update actually applied
  this step (`-lr * u`).
- `track_param_*` — scalar norms of the weight **after** this step's update. All 4 paths use a
  cheap **derived pseudo-value** (`pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)`, mirroring
  the real apply formula) rather than a second real read, because the true pre-update `w` is needed
  in-scope for gram metrics (see below) — `step_experts` didn't compute this until it also needed
  `pseudo_w` as gram's `W_after` (see "Norm/gram/spectrum tracking" → gram below); now all 4 paths
  are consistent.
- `track_spectrum_*` — raw singular-value vectors (`update`/`param`), consumed by
  `spectrum_logging.py`.
- `track_gram_*` — functions of `(W_before, V_raw, W_after)` triples (`gram_helper.
  calculate_gram_metrics`): `W_before` is the pre-update weight, `V_raw` is the **raw** effective
  grad/momentum (whatever's fed into `self.lmo()` -- see "V_raw is the raw moment, not the LMO
  update" below), and `W_after` is `pseudo_w` (the same post-update approximation `track_param_*`
  already uses, not a fresh real read). `U = W_after - W_before` (the exact realised displacement)
  and `A = -U` are derived internally -- see `gram_helper.py`'s module docstring and
  `gram_matrix.md` for the full formula catalogue and the "which tensor answers which question"
  framing. Gated by a single cumulative `self.gram_level: int` (0 = off, no-op; 1/2/3 = increasingly
  expensive, each level includes all lower levels), set alongside `norms_to_log` via
  `calculate_norm_at_next_step(norms_to_log, gram_level)`, driven by `config.metrics.gram_level`
  (mirrors `config.metrics.norms_to_log`/`log_norm_freq` exactly -- same cadence, no separate gate).
  `calculate_gram_metrics` returns a mix of 0-d scalar and 1-d vector tensors (fixed key SET per
  level, independent of parameter shape -- disco.py's DDP/FSDP/experts packing code relies on this;
  ~121 keys at level 3, comparing 4 tensors -- `W_before`, `W_after`, `V_raw`, `U` -- pairwise).
  Every call site calls `calculate_gram_metrics` unconditionally (cheap no-op at `gram_level=0` --
  returns `{}` instantly) and only branches on the result's truthiness where required (e.g. before
  `torch.stack(...)` for a collective) -- there is deliberately no separate "is gram active" flag
  anywhere. Vector-valued entries get popped and rendered as W&B atlas-grid images (index vs. value
  line plots, one grid per metric name, gated by `config.optimizer.enable_gram_plot`) and/or
  exported to a Parquet file uploaded as a W&B Artifact (gated by
  `config.optimizer.enable_gram_export`, mirrors `spectrum_logging._export_spectrum` exactly), by
  `gram_vector_logging.py` before reaching a scalar logger (copy-and-adapted from the grid/layout
  machinery `spectrum_logging.py` provides for `track_spectrum_*`, kept as a separate module -- see
  that file's docstring for why). A small, opt-in set of metric names (currently just `V_R_raw`,
  see `_MEAN_MIN_MAX_METRIC_NAMES`) also get 3 cheap derived scalars unconditionally
  (`..._mean`/`..._min`/`..._max`, e.g. `track_gram_V_R_raw_mean/...`). Scalars otherwise need no
  handling at all, they're already valid logger values.

### `norm_helper.py` cost controls

Three things in the norm pass were doing avoidable work.

**The full spectrum was computed, packed, all-gathered, copied to CPU, then thrown away.**
`optimizers/spectrum_logging.py` is the only consumer, and `process_norms_for_logging` pops every
`track_spectrum_*` entry and returns early unless `enable_spectrum_plot` or
`enable_spectrum_export` is set -- both default to `False`. `calculate_norm` now takes
`want_spectrum`, threaded from those two flags through the existing
`calculate_norm_at_next_step` seam (`AbstractDiSCO.track_spectrum`). When it is off, the
per-family spectrum buffers are never allocated, `_pack_segments` drops the segment, and the
unpack side skips it (it was already keyed on `if "upd_spec" in offsets`). Measured on a 157M
dense model with `norms_to_log=all`: the spectrum was **91.8%** of the gather payload (900 KiB of
980 KiB per rank per logging step), so the all-gather shrinks ~12x -- 245 MiB -> 20 MiB at
world_size 256. No logged value changes, because those tensors never reached a logger anyway.

`sigma_max` is handed out as its own key rather than being read off `spectrum[0]`, so radial keeps
its free leading singular value when spectrum logging is off. `disco.py:_pop_spectrum` is the one
place that takes both keys out of a `calculate_norm` result -- they must both be removed before
the caller iterates `.values()`, since the flat buffers are sized to `norms_to_log` exactly.

**Requested norms now decide whether the SVD runs at all.** `fused_metrics` always called
`svdvals`. There are now three tiers, selected in `calculate_norm`: no SVD when the requested
norms are all in `_SVD_FREE_NORMS` (`l1_to_rms`, `rms_to_inf`, `supremum`, `frobenius_norm`,
`average_entry_size`) and no spectrum is wanted; sigma_max only (`rms_to_rms`, `stable_rank`); and
the full spectrum (`condition_number`, `effective_rank`, `effective_rank_squared`). Note that
`condition_number` is in the `"default"` norm set, so most runs still land in the last tier --
trimming `norms_to_log` is what unlocks this.

**1-D parameters no longer build a `d x d` matrix.** `calculate_norm` used to expand any 1-D
parameter into `torch.diag_embed(v)` and run a full SVD over it, purely so the operator norms
would be defined. Since only `numel() == 1` params are routed to `step_scalar`, every affine norm
weight `[d_model]` took that path -- twice per logging step. `diag_metrics` computes the identical
values in closed form from `sort(|v|)`: for `diag(v)` every row and column has one non-zero entry,
so `row_l2 == col_l2 == |v|`, the singular values are `sort(|v|, descending)`, and
`fan_in == fan_out` makes `rms_to_rms`'s scale exactly 1. Measured at d=4096: **676 ms -> 0.49 ms**,
and a 64 MiB fp32 allocation avoided. `average_entry_size` deliberately keeps dividing by `d`
(i.e. averaging over the `d^2` embedded entries, `d^2 - d` of them structurally zero) to preserve
continuity of existing series.

*Caveat, measured not assumed:* the `opt_moe` flavors in this repo have **no 1-D trainable
parameters** at all (their norms are non-affine), so this particular win is latent for them --
it matters for any model with affine norm weights, and it removes an O(d^3) trap either way.

**cuSOLVER driver: chosen per use, not globally.** Measured on GH200 / torch 2.12 at the exact
matrix shapes of the 7B/8B and 30B MoE flavors (`dim=2048`, so every tracked matrix is tall or
wide, none square):

| matrix | gesvd | gesvdj | gesvda |
|---|---|---|---|
| attn `wq` `[4096,2048]` | 158 ms | 181 ms | **26 ms** |
| ffn `w1` `[6144,2048]` (qwen30b) | 191 ms | 187 ms | **26 ms** |
| moe expert `[768,2048]` (qwen30b, x128/layer) | 37 ms | 35 ms | **7 ms** |
| embed / lm_head `[201088,2048]` | 292 ms | 393 ms | **85 ms** |

`gesvdj` -- the obvious "Jacobi is faster" switch -- is **not** faster here, sometimes slower.
`gesvda` is 4-7x faster than the historical `gesvd` at every shape. But it is an *approximate*
driver and must not be the blanket default:

- `sigma_max` is exact -- 0 to ~1e-7 relative on every shape tested.
- `sigma_min` is not: ~1e-2 relative on ill-conditioned input, and 100-300% on rank-deficient
  input. That lands straight on `condition_number` (`S[0]/S[-1]`), measured 65-77% wrong for
  rank-deficient matrices.
- It can fail outright: `_LinAlgError: the algorithm failed to converge` on a rank-deficient
  2048x2048 input. That would take down a training run.

So `_svdvals(W, sigma_only=True)` uses `gesvda` **only** where nothing reads past `S[0]`, and
falls back to `gesvd` if it raises; everything that touches the whole spectrum keeps `gesvd`.
`fused_metrics_sigma_only` is the tier that gets it, selected when `norms_to_log` stays inside
`_SVD_FREE_NORMS | _SIGMA_ONLY_NORMS`. Overridable via `DISCO_SVDVALS_DRIVER` and
`DISCO_SVDVALS_DRIVER_SIGMA_ONLY`.

End-to-end `calculate_norm` cost, and why trimming `norms_to_log` is worth real money:

| matrix | default set (incl. `condition_number`) | drop `condition_number` | drop all SVD norms |
|---|---|---|---|
| attn `wq` `[4096,2048]` | 364 ms | 28 ms (**13x**) | 0.26 ms (1378x) |
| ffn `w1` `[6144,2048]` | 371 ms | 28 ms (**13x**) | 0.12 ms (3090x) |
| moe expert `[768,2048]` | 83 ms | 8 ms (**10x**) | 0.25 ms (329x) |
| embed `[201088,2048]` | 576 ms | 89 ms (**6.5x**) | 2.0 ms (283x) |

`condition_number` and `effective_rank*` are the only reason the accurate driver is needed, and
`condition_number` is in the `"default"` norm set -- so most runs pay ~10x more than they need to.

**`torch.compile` on `fused_metrics` cost more than the SVD it wrapped -- removed.**
`torch.linalg.svdvals` cannot be lowered by inductor, so the decorator graph-broke around the one
expensive operation and roughly doubled the total. Identical math, eager vs compiled:

| matrix | eager | compiled |
|---|---|---|
| `[768,2048]` | **40.1 ms** | 83.1 ms |
| `[4096,2048]` | **177.8 ms** | 360.4 ms |
| `[6144,2048]` | **180.8 ms** | 381.7 ms |
| `[201088,2048]` | **302.4 ms** | 576.3 ms |

The non-SVD arithmetic in there is ~0.06 ms of reductions, so there was never much for the
compiler to win. `fused_metrics` and `fused_metrics_sigma_only` are now eager;
`fused_metrics_no_svd` keeps its decorator, since it has no decomposition to break on and compile
genuinely helps it at large shapes (2.10 ms vs 3.20 ms on the embedding). **This is a ~2x on every
norm call in every `step_*` path and needs no config change.**

**The gathered buffer must be materialised before unpacking.** `funcol.all_gather_tensor`
returns an `AsyncCollectiveTensor`, a tensor subclass that routes every operation through
`__torch_dispatch__` so it can insert the wait. All three logging gathers (`step_experts`,
`step_ddp`, `_gather_and_log_fsdp`) then index that buffer once per (parameter, expert, metric)
to build the metrics dict -- about 194k times per logging step for a 600M MoE -- and every one of
those was paying Python-level subclass dispatch.

Measured, 193,536 index operations on the same buffer:

| | time |
|---|---|
| raw `AsyncCollectiveTensor` (what the code did) | 6.93 s |
| after `.wait()` | **0.28 s** (25x) |
| after one bulk `.cpu()` | 0.02 s |

`_materialize_gathered` calls `.wait()` once, up front. This is semantically free -- the wait has
to happen before the first read regardless; it just stops every *subsequent* read going through
the slow path. It was worth ~8.7 s of a 12.2 s `step_experts` logging pass, which is more than
every other optimization in this file put together, and it was invisible to reasoning: it showed
up only after instrumenting a real run and then eliminating each candidate by measurement.

**What actually moved the needle, measured end to end.** Several optimizations were made here;
only two of them show up in a real run, and it is worth recording which, because the
microbenchmarks were misleading on their own. Wall time of one logging step on a 4-GPU MoE run
(M3-moe-600M, 196 fsdp + 84 expert + 30 embed params, `norms_to_log=all`, `log_norm_freq=2`):

| state | logging step |
|---|---|
| before this work | **131 s** |
| + batched expert norms, + `torch.compile` removed | 130 s |
| + float64 Gram spectrum | 39.7 s |
| + `polar_express_triton` `steps` fix | 38.7 s |
| + `_materialize_gathered` (`.wait()` on the gathered buffer) | **13.0 s** |

**10x overall**, with every metric kept and accuracy improved. The two that mattered are the Gram
spectrum and the `.wait()`; the `.wait()` alone saved more wall time (25.7 s) than it saved inside
`optimizer.step` (8.3 s), because the ~260k metric values also stop being `AsyncCollectiveTensor`
views on their way through `process_norms_for_logging` to the logger.

Inside `optimizer.step`, instrumented: `step_experts` 12.2 s -> 4.2 s, `step_fsdp` 1.07 -> 0.72 s,
total 13.6 -> 5.2 s. `calculate_radial_metrics` is now 65% of what remains (2.7 s, one call per
expert matrix) and is the obvious next target -- it batches naturally over the expert axis, since
expert `radial_state` is already shaped `(num_local_experts,)`.

The two that did *not* matter, and why:

- **Batching** buys nothing while `norms_to_log` includes `condition_number`, because that forces
  the full-spectrum path and `gesvd` serialises internally -- batched and looped `gesvd` measure
  the same. It pays off once the spectrum comes from the Gram (batched `eigvalsh` does scale).
- **Removing `torch.compile`** is a real 2x on `fused_metrics`, but with `norms_to_log=all` most
  parameters (the 5376 expert matrices here) go through `calculate_norm_batched`, which never used
  `fused_metrics`. It touches only the ~56 dense params a rank owns -- about 2.4 s of a 130 s
  step, inside the noise.

The other two are not wrong, they are just nearly inert *in this configuration*, and the reasons
are worth knowing:

- **Batching** buys nothing while `norms_to_log` includes `condition_number`, because that forces
  the full-spectrum path and `gesvd` serialises internally -- batched and looped `gesvd` measure
  the same. It pays off once the spectrum comes from the Gram (batched `eigvalsh` does scale) or
  in the sigma-only tier via `gesvda`.
- **Removing `torch.compile`** is a real 2x on `fused_metrics`, but with `norms_to_log=all` the
  bulk of the parameters (the 5376 expert matrices here) go through `calculate_norm_batched`,
  which never used `fused_metrics` at all. It only touches the ~56 dense params a rank owns --
  about 2.4 s of a 130 s step, i.e. inside the noise.

**The full spectrum now comes from a float64 Gram matrix, not `svdvals`.** This is the change
that makes keeping `condition_number` and `effective_rank*` affordable, rather than trading them
away. `_gram_spectrum` computes `sqrt(eigvalsh(W W^T))` in float64, orienting so the Gram is over
the smaller dimension and chunking the reduction for very tall matrices. It is both **faster and
more accurate** than the fp32 `svdvals` it replaces -- relative error on `sigma_min` /
`condition_number` against a float64 SVD, `[768,2048]`:

| conditioning | fp32 `svdvals` | fp64 Gram |
|---|---|---|
| well-conditioned | 4.9e-08 | **1.9e-13** |
| kappa 1e3 | 2.8e-07 | **2.7e-11** |
| kappa 1e6 | 3.0e-04 | **2.2e-05** |
| kappa 1e8 | 3.9e-02 | **7.0e-03** |
| rank-deficient | 9.4e-01 | **3.6e-02** |

*float64 cannot be traded for a normalization trick.* Rescaling `W` by its Frobenius norm, by its
max-abs entry, or rescaling the Gram before `eigvalsh` were all measured and all change nothing --
`sigma_min` error stays ~1e-4 at kappa 1e2, ~1e-2 at kappa 1e3, and total at kappa 1e6 for every
variant. `kappa(W W^T) = kappa(W)^2` is scale-invariant, so normalization only guards against
overflow, which is not the failure mode.

*And float64 is not the cost you would expect.* The Gram GEMM is **faster** in fp64 than in true
fp32 here (0.79-0.82x, CUDA-event timed): on Hopper/GH200 FP64 has dedicated tensor cores while
non-TF32 FP32 does not and falls back to CUDA cores -- 49.7 TFLOPS fp32 vs 61.8 TFLOPS fp64 on a
68.7 GFLOP Gram. (TF32 reaches 434.8 and bf16 841.1, but neither is usable against a product that
squares the condition number; fp64 also makes this immune to a globally enabled TF32 flag.) And
the GEMM is not the cost anyway -- `eigvalsh` on the `[k,k]` Gram dominates and is only 1.03-1.07x
slower in fp64.

`DISCO_SPECTRUM_BACKEND=svd` restores the old path for continuity with an in-flight run. Note this
*is* a numerics change: every spectrum-derived metric moves, in the direction of correctness.

**Net effect with `norms_to_log` untouched** (all 10 norms, nothing dropped), against the state at
the start of this work (`torch.compile`d `fused_metrics` + fp32 `gesvd`, one matrix at a time):

| matrix | before | now | |
|---|---|---|---|
| moe expert `[768,2048]` | 83.2 ms | 8.0 ms | 10.4x |
| attn `wq` `[4096,2048]` | 363.8 ms | 25.7 ms | 14.1x |
| ffn `w1` `[6144,2048]` | 371.0 ms | 25.8 ms | 14.4x |
| embed `[201088,2048]` | 576.3 ms | 57.4 ms | 10.0x |

and for the 282 expert matrices one rank owns for qwen30b-a3b at EP=64: **23.5 s -> 0.27 s (87x)**.

**Batched expert norms.** `step_experts` used to call `calculate_norm` once per expert.
`_batched_expert_norms` now computes them all up front, grouped by shape and chunked, via
`calculate_norm_batched`; the consuming loop keeps its exact previous structure and packing order,
with the two calls replaced by lookups.

The batch must be formed **across layers**, not across a parameter's expert axis. Real runs put
1-2 local experts on a rank (128 experts over 64-128 GPUs), and at that size batching within a
param is worthless -- measured **1.00x at E=1 and 0.56x at E=2**, i.e. actively worse. Every
expert matrix in the model shares one of two shapes, so grouping across all blocks gives batches
in the hundreds instead. Chunked at `_GESVDA_MAX_BATCH=128`: a batch of 282 raised
`CUSOLVER_STATUS_INVALID_VALUE` while 192 worked, and 128 is already within 5% of the best
observed throughput.

Where that leaves qwen30b-a3b at EP=64 -- 47 MoE layers x 3 matrices x 2 local experts = 282
matrices of `[768,2048]` per rank per logging step:

| | time | vs before |
|---|---|---|
| before this pass (compiled, per-matrix, all norms) | 23.5 s | 1x |
| now, per-matrix, all norms | **11.0 s** | 2.1x |
| now, batched, all norms | 11.4 s | 2.1x |
| now, per-matrix, sigma-only tier | 2.2 s | 10.9x |
| now, batched, sigma-only tier | **0.33 s** | **71x** |

Note batching buys nothing on its own for the full-spectrum tier -- `gesvd` serialises internally.
The large win needs `norms_to_log` to drop `condition_number` and `effective_rank*`; with them in
(the `"default"` set, and `"all"`), the 2.1x from removing `torch.compile` is what you get.

**gram and radial were checked for the same problems and do not have them.** Neither is
`torch.compile`d, so neither pays the graph-break tax. gram's cost is genuine linear algebra (11
`torch.linalg` calls -- `eigh`/`svd` on Gram factors), measured 6.7 ms/matrix at level 1 and
76.9 ms at level 2; it is gated behind `gram_level` and off by default. radial is 1.5 ms/matrix,
of which the radiality/aus axis reductions this pass added are only 0.09 ms -- the rest is CPU
dispatch across the many small 0-d ops that 26 metrics require. Compiling
`calculate_radial_metrics` was measured at **1.84x** (1.53 -> 0.83 ms) with the accumulators
bit-identical, and is *not* enabled: it saves ~0.2 s against an ~11 s budget, and it perturbs
`angle_from_cos`. That perturbation is not a regression -- `angle_from_cos` uses `arccos`, which
is already 14% wrong at a 1e-3 relative step in eager (which is exactly why `angle` is computed
via `atan2` and is bit-identical under both).

### `step_experts` update-norm scale (behaviour change)

`step_experts` measured `calculate_norm(u[ep_idx])` -- the bare LMO output -- while
`step_embedding`, `step_ddp` and `step_fsdp` all measure `-lr * u`. Expert `track_update_*` series
were therefore off by a factor of `lr` relative to every other family. Now fixed to `-lr * u`.
**This changes the value of existing expert update-norm series**; it is a correction, not a
refactor, and comparisons against runs from before it will show a step change of exactly `lr`.

### `V_raw` is the raw moment, not the LMO update

`calculate_gram_metrics(W_before, V_raw, W_after, ...)`'s `V_raw` argument is the **raw** effective
grad/momentum -- the exact tensor about to be passed into `self.lmo()` -- not the LMO-processed
update `u = self.lmo(...)` that every path already computes for the real parameter update. This
matches the "study optimizer-state geometry" framing in `gram_matrix.md` (as opposed to "study
weight dynamics", which is what the *realised displacement* `U = W_after - W_before` -- derived
internally inside `gram_helper.py`, not the same `U` as `self.lmo()`'s `u` -- is for).

Every path already computes the raw grad immediately before calling `self.lmo()`, so capturing it
for gram is "keep one more reference alive a little longer", not new compute or communication --
mirrors the existing `u_keepalive`/`pseudo_w` pattern (see "simultaneity" below) exactly, just one
step earlier:
- `step_embedding`: the raw grad `g` is already in scope right where `u = self.lmo(g, ...)` is
  called, a few lines before the gram call -- no new variable needed, just pass `g` instead of `u`.
- `step_ddp`: `lmo_inputs` (from `_prepare_ddp_lmo`, Phase A) already holds the raw per-owned-index
  grad and is never mutated afterward -- reused directly at the later gram call site.
- `step_experts`: a new `all_raw_grads` list, populated from `big_g` (captured right before
  `self.lmo(big_g, ...)`) alongside the existing `all_updates` list.
- `step_fsdp`: a new `g_keepalive` list, populated alongside `u_keepalive` in both the fast path
  (aliases the step-persistent `full_g_bufs[bucket_idx]` workspace buffer -- free) and the slow path
  (keeps a second reference to the freshly `torch.cat`'d `full_g`, which would otherwise be
  discarded once the loop moves to the next bucket).

Verified `AbstractDiSCO.lmo()` never mutates its input tensor in place (every backend, eager and
Triton, only ever rebinds to new tensors or writes into separately-allocated `out=` buffers) --
aliasing the raw-grad reference this way is safe; it will always reflect the true pre-LMO value.

### `W_before`/`W_after`/`V_raw` simultaneity (why all 4 paths compute `pseudo_w`)

`calculate_gram_metrics` needs the pre-update weight, its raw moment, and the post-update weight
simultaneously in scope. Historically `track_param_*` was computed *after* the real update was
applied, so the pre-update weight was already out of scope. Fixed by reordering — moving the point
where the real update gets applied to run *after* norm/gram calculation instead of before, in
`step_embedding`/`step_ddp`/`step_fsdp` (`step_experts` already computed weight-norm pre-update, no
reordering needed there). Critically, **no collective changed** in any of these — only the order of
two already-existing blocks. The update needed a small amount of extra lifetime in some paths
(`step_fsdp` needed a new `u_keepalive` list, now also `g_keepalive`; `step_ddp`'s
`local_updates`/`lmo_inputs` were already alive long enough) — "a bit more temporary memory held a
little longer," not new communication.

All 4 paths pass `pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)` as gram's `W_after` (the same
tensor already used for `track_param_*`). `step_experts` didn't compute `pseudo_w` at all until the
3-tensor gram spec needed it — it used the raw pre-update `p_local[ep_idx]` for both weight-norm and
gram directly. Fixed to fetch `lr`/`wd` once per block (`self.groups_info[self._expert_block_group_idx[block_idx]]`)
and compute `pseudo_w` per-expert, same formula as the other 3 paths.

**Real bug found and fixed here**: when first reordering `step_ddp`'s weight-norm loop, an
`if u is None: continue` was copied from the neighboring update-norm loop. That's wrong for
weight-norm specifically: `local_updates[my_idx] is None` (no gradient this step) does not mean
the real apply skips the param — Phase B substitutes a zero update (`zero_by_shape`) and weight
decay still applies. Skipping would have silently zeroed `track_param_*` for any DDP param lacking
a gradient on a logging step. **Why can `u` be `None` at all?** `p.grad is None` happens in
ordinary situations — gradient accumulation boundaries, pipeline-parallel stages not executed this
micro-batch, `requires_grad=False` params still tracked, or a param genuinely untouched by this
step's forward/backward. Fix: substitute `torch.zeros_like(w)` and let the computation proceed
normally, matching what the real apply does, rather than skipping the param.

### Communication fusion

`step_experts` already fused everything (update/weight/gram norms + both spectrum halves) into one
`torch.cat` + one `all_gather_tensor`. `step_ddp`'s Phase D and `step_fsdp` used to issue up to 5
separate `all_gather_tensor` calls each; both now use the same pattern via a shared helper,
`_pack_segments(segments) -> (buffer, offsets)`, which derives offsets from what was *actually*
packed rather than hand-computing them (hand-derived offsets are exactly the kind of thing that
caused the `step_ddp` bug above) — see `_pack_segments`'s docstring and the `_gather_and_log_fsdp`
method for the pattern.

### If you're adding a real gram metric formula

`gram_helper.py` is a cumulative, per-level design (mirrors `norm_helper.fused_metrics`'s
shared-computation pattern): `_build_gram_core` builds every self-/cross-Gram and correlation matrix
once (`G_Wm`/`G_Wp`/`G_V`/`G_U`, `C_Wm`/`C_Wp`/`C_V`/`C_U`, `C_WV`/`C_WU`/`C_VA` — `Wm`/`Wp` = weight
before/after, `V`/`U` = raw momentum / realised displacement), and
`_level1_metrics`/`_level2_metrics`/`_level3_metrics` derive that level's metrics from shared state
(level N is cumulative with levels < N). There's no per-metric registry (`GRAM_METRIC_FUNCTIONS` is
gone).

1. Add the computation inside the right `_level{1,2,3}_metrics` function (or extend `_GramCore`/
   `_Level2Extras` if it needs new shared intermediates).
2. Add its name to the matching level in `GRAM_SCALAR_NAMES_BY_LEVEL` (0-d output) or
   `GRAM_VECTOR_NAMES_BY_LEVEL` (1-d output) — these two dicts are the source of truth for the
   fixed, shape-independent key set `calculate_gram_metrics(..., level=L)` returns; every vector is
   length `m` (see `gram_helper.py`'s module docstring — NOT `min(m, n)`, don't reuse
   `norm_helper`'s spectrum-length tables for gram vectors). Naming convention:
   `{tensor_prefix}_{field}` (`V`/`U`/`Wm`/`Wp` for the 4 self-Gram tensors, `VA`/`WV`/`WU` for the 3
   cross-Gram pairs, `G_*`/`C_*` for eigenspectra, `K_V`/`K_U`/`J`/`Q_*` for level-3 whitened
   quantities) — keep new metrics consistent with this so `gram_vector_logging.py`'s per-metric-name
   atlas grids stay readable.
3. No `disco.py` call-site changes needed for scalars. A **new vector-valued** metric needs its
   contribution counted in the three `_precompute_*_gram_vector_metadata` methods' `n_vec =
   len(self.gram_vector_names)` — already automatic, since those methods re-read
   `self.gram_vector_names` fresh each time they run, but double check the offset math if you
   change a vector's *length formula* rather than just adding another same-length vector.
4. Decide what your formula should do when `W_after == W_before` (the `step_ddp` zero-gradient case
   feeds `u = torch.zeros_like(w)`, so `pseudo_w == w`, hence `U_actual = 0` inside
   `calculate_gram_metrics`) — verified all existing level 1-3 formulas degrade gracefully to
   finite, `eps`-guarded values for this case (traced through by hand + covered in the standalone
   verification script); handle any new formula's zero-input behavior explicitly rather than relying
   on incidental float behavior.
5. Vector metrics automatically get their own atlas grid (dense + MoE) via
   `gram_vector_logging.py` — no changes needed there either, since it discovers grid names
   dynamically from tracked key names (see that file's docstring).

### `step_embedding` gram tracking: large-vocab OOM fixed, cost-gated behind `DISCO_TRACK_EMBED_GRAM`

`gram_helper._gram(X) = X @ X.T` forms an `m x m` matrix, where `m` is the row-count after
orientation. `step_embedding`'s `embed_params` includes the `output`/lm_head weight
(`[vocab_size, hidden_dim]`) alongside `tok_embeddings` (same shape) -- previously, `need_T =
CONST_NAME_OF_EMBEDDING in p_name` only transposed for params literally named `"tok_embeddings"`,
not `"output"`, so the lm_head weight's `m` stayed at `vocab_size` instead of being reduced to
`hidden_dim` -- at a ~200k vocab this was a `[200_000, 200_000]` fp32 matrix (~160GB), an immediate
CUDA OOM (hit in practice, not hypothetical).

**Fixed**: `calculate_gram_metrics` now decides orientation unconditionally from shape (`rows <=
cols`, ignoring any caller-supplied `transpose`/`need_T` -- see `gram_helper.py`'s module docstring
and `calculate_gram_metrics`'s own docstring), so both `tok_embeddings` and `output` always get
`m = hidden_dim`, never `vocab_size`. This is a deliberate, session-wide semantic choice (not
special-cased for embeddings): any parameter with `D_out > D_in` gets the same treatment, e.g. FFN
up-projections now track input-feature-wise dynamics instead of output-channel-wise.

**Still gated, though, on cost rather than correctness**: `tok_embeddings`/`output`'s gram
computation remains far more expensive than any other tracked param, even without the OOM --
`_build_gram_core` materializes several full-size `[hidden_dim, vocab_size]` copies (row-normalized
factors, `U_actual`, `A_actual`), which at a ~200k vocab and multi-thousand hidden dim is on the
order of tens of GB of transient memory and multiple seconds of forced-fp32 GEMM (the small
`[hidden, hidden]` Gram matrix itself is cheap; building it from the full-size factors is not) --
*per parameter, per logging event*, for exactly these two parameters. `calculate_norm_at_next_step`
already lets you tune `gram_level` (and hence gram tracking's cost) per step; `DISCO_TRACK_EMBED_GRAM`
(default `"1"`, or set `optimizer.track_embed_gram = False` directly at runtime) is a second,
independent switch specifically for these two expensive params, so you can e.g. run `gram_level > 0`
every step for cheap layers while only enabling embed/output gram at sparse checkpoints. `step_ddp`/
`step_fsdp`/`step_experts` are unaffected either way (never touch vocab-scale dimensions).

One conceptual note worth keeping in mind when reading `tok_embeddings`'s (as opposed to `output`'s)
gram metrics: `tok_embeddings` is a lookup table, not a jointly-computed Linear layer -- each row's
gradient depends only on whether that token appeared in the batch, with no forward-pass coupling
between different vocab rows. Transposing doesn't change that underlying gradient structure, but it
does change what the resulting `[hidden, hidden]` Gram matrix answers: `G[a, b] = Σ_j
tok_embeddings[j, a] · tok_embeddings[j, b]`, summed over the whole vocabulary, asks about
correlation/redundancy *between hidden dimensions* across the embedding table (a real, studied
quantity -- embedding anisotropy/dimension collapse), not "is this output channel dominant" in the
sense the same metric name means for e.g. `attention.wq`. `output`/lm_head doesn't have this caveat
-- it's a genuine jointly-computed Linear layer, so its transposed (input-feature-wise) reading is
exactly as valid as any other `D_out > D_in` layer's.

### LMO backends: `steps` was silently ignored, and what to set it to

`polar_express_triton` iterated its whole coefficient table (`for a, b, c in coeffs`) rather than
`for k in range(steps)`. That table is 8 entries long regardless of the argument, so
**`backend_steps=5` ran 8 Newton-Schulz iterations on that path** while
`muon_utils.zeropower_via_polar_express` ran 5 for the same setting. The two "polar express"
backends were not the same computation, and the config knob did nothing here. Measured before the
fix: error against the exact polar factor was a flat 0.0064 for every `steps` from 3 to 12. Now
fixed (`coeffs = coeffs[:steps]`).

`newton_schulz_triton` does **not** have this bug -- it honours `steps` correctly. It does,
however, special-case `steps == 5` to a different (much better) coefficient set than
`zeropower_via_newtonschulz5` uses: at `steps=5` the two measure 0.0169 and 0.1707 respectively,
a 10x quality gap between two backends with the same name. Left alone here, but worth knowing if
anything selects `newtonschulz5`.

**What to set `backend_steps` to.** After the fix, quality and cost of `polar_express_triton` on
the two batched calls `step_experts` makes per optimizer step (qwen30b-a3b at EP=64):

| steps | error vs `UV^T` | expert LMO / step |
|---|---|---|
| 4 | 0.3639 | 13.22 ms |
| 5 | 0.0899 | 15.96 ms |
| **6** | **0.0061** | **18.73 ms** |
| 7 | 0.0061 | 21.54 ms |
| 8 (the old silent behaviour) | 0.0064 | 24.36 ms |

There is a sharp knee between 5 and 6, and nothing to gain past 6. **`backend_steps=6` reproduces
the orthogonalization quality runs were already getting, 1.30x cheaper.** Note the flip side: with
the bug fixed, leaving `backend_steps=5` is a real change to the optimizer -- 14x less orthogonal
than what those runs actually did -- so it should be an explicit choice, not a default inherited
from a config written when the knob was inert.

### `gram_polar_express` (opt-in, off by default)

`gram_newton_schulz.py` vendors the Gram Newton-Schulz iteration from
https://github.com/Dao-AILab/gram-newton-schulz (MIT), which iterates on the small symmetric
`X X^T` instead of on `X`. Registered as `gram_polar_express`; the default backend is unchanged.
At 8 iterations it is both **2.3x more accurate and 1.21x faster** than `polar_express_triton`
(0.0028 vs 0.0064, 20.17 vs 24.31 ms/step) -- but only with the `quack` symmetric-GEMM kernels;
with the torch fallback it is slower than the default and not worth enabling.

Two cautions. It runs in float16, not bfloat16: the Gram iteration squares its operand each step
so mantissa width matters more than range (measured 0.0027 fp16 vs 0.0132 bf16 vs 0.0017 fp32,
fp32 being ~8x slower). And passing quack's kernels an unsupported dtype **segfaults the process**
rather than raising, so the module refuses anything but the half dtypes rather than discovering
that mid-run.

Unlike everything in norm_helper/radial_helper, this sits on the *update path* -- its output is
the direction the optimizer applies -- so enabling it changes the training trajectory.

### Per-rank metric logging (`metrics.save_all_shard_ranks`, off by default)

Each shard rank already computes metrics for a disjoint subset of the parameters; the logging
all_gather exists only to bring them to one rank. With `save_all_shard_ranks` every shard rank
keeps its own instead, and all three gathers (`step_experts`, `step_ddp`,
`_gather_and_log_fsdp`) are skipped. Each unpack loop then skips parameters it does not own
(`owner_rank != rank`) and reads at offset 0 rather than `owner_rank * per_rank_total`.

The point is not the collective -- that is a few MB -- it is that one rank stops building the
metrics dict for the *whole model*: for qwen30b-a3b that is 845,664 entries per logging step on
one rank, versus 13,213 per rank across 64 shards.

**It only engages when the metrics side really does log on every shard rank.** The predicate is a
conjunction, evaluated in `trainer.py`:

| `save_all_shard_ranks` | `save_for_all_ranks` | logger enabled | result |
|---|---|---|---|
| False | any | any | gather (unchanged) |
| True | False | any | **raises `ValueError`** |
| True | True | False | gather (unchanged) |
| True | True | True | local, no gather |

`save_all_shard_ranks` alone raises rather than being ignored, because it is applied inside
`_build_metric_logger` only after `should_log` is already true, and with
`save_for_all_ranks=False` that has been narrowed to one global rank -- so acting on the flag
would have every rank skip the gather while a single rank logs its own 1/N and the rest vanished
silently.

Two further constraints, both learned the hard way:

- **The decision must come from config, never from a per-rank predicate.** The gather is a
  collective: if some ranks skipped it and others did not, the job would *hang* rather than
  misreport. Config values are identical on every rank, so uniformity is structural.
- **Widening the unpack gate is not enough.** `norms_at_current_step.update()` sits behind a
  second gate (`_stores_norms`, formerly `is_dp_rank_0`). Leaving that one narrow made ranks 1-3
  compute their whole share and discard it -- 40 logged keys instead of ~65,000. Both gates have
  to move together.

Cost: one W&B run per shard rank (`base_log_dir` already appends `rank_{n}`), so reassembling a
full picture means reading N runs.

### Exact top singular pair, and why `power_iteration.py` no longer iterates

radial's radiality metrics need the leading singular *vectors* of the **pre-update** weight, which
`calculate_norm` never sees (it is called on `-lr*u` and on `pseudo_w`). That was originally going
to be a warm-started power iteration, on the assumption that a decomposition of `W_before` would
be prohibitive.

The Gram work removed that assumption. `_gram_spectrum` already forms `W W^T` in float64 and calls
`eigvalsh`; asking for `eigh` instead returns the eigenvectors of the same matrix for **1.05-1.07x**,
and the leading one *is* `u1`, with `v1 = W^T u1 / sigma` in one matvec. So
`gram_top_singular_pair` is exact -- sigma error 5e-14 to 6e-13, `u1` residual ~1e-15 -- where the
iteration was landing at 1e-2 to 1e-3 cold, and it needs no warm-start state carried between
logging steps. `power_iteration.IS_STUB` is now `False` and the module delegates to it; the
iteration-shaped arguments (`v0`, `n_iter`, `tol`, ...) are accepted and ignored.

Batched, the marginal cost of the vectors is +47 ms on 1344 `[384,1024]` and +32 ms on 282
`[768,2048]`. `W_before` still needs its own decomposition, ~+0.3 s on a `step_experts` of 4.2 s,
which is why it is computed in shape-grouped batches inside `_batched_expert_norms` rather than
once per expert (that would be ~4.6 s).

This is what took the radial family from 21 live metrics + 5 hard zeros to **26 live**:
`spectral_radius`, `spectral_growth`, `spectral_relative_step`, `radiality_rms_to_rms` and
`aus_rms_to_rms` are now real values.

### Radial-dynamics metrics (`radial_helper.py`) -- always-on, independent of gram/norm config

A third tracked-metric family, alongside norm/spectrum and gram, living in its own module
(`radial_helper.py`). Unlike gram (row-wise Gram-matrix framework, needs `V_raw`, gated behind
`gram_level`), radial metrics ask a different, simpler question: how does a weight's *whole-tensor*
norm and direction evolve step to step. Every metric is a single Frobenius-norm-scale scalar (never a
row-wise vector), computed from just `W_before`/`W_after` (the same pair already used as gram's
`W_before`/`W_after`) -- cheap enough that `calculate_radial_metrics` is called **unconditionally**
whenever any per-param logging fires at all, independent of `gram_level` and `norms_to_log`'s
contents, at all 4 `step_*` call sites (including `step_embedding`, ungated by
`track_embed_gram` -- radial's cost is O(1) scalars, not O(vocab_size) Gram matrices).

Four running accumulators (`raw_A2`, `angular_A1`, `angular_A2`, `R1`) track the parameter's
cumulative history and persist across checkpoint save/restore -- stored in
`self.state[p]["radial_state"]` (the same place `momentum_buffer` lives), included automatically in
`torch.optim.Optimizer`'s default `state_dict()`/`load_state_dict()`, no extra plumbing. 3-D (expert)
params get one accumulator set *per expert index* (each expert has its own `W_before`/`W_after`
pair), shape `(num_local_experts,)`.

**Ordering: norm -> radial -> gram.** All four `step_*` paths run the three families in that
order (previously norm -> gram -> radial). Radial sits in the middle because it *consumes* output
of the norm pass: `calculate_norm` already returns the full descending `spectrum` for the update
`-lr*u` and for the post-update weight `pseudo_w`, so `spectrum[0]` hands radial `sigma_max` of
both for the cost of an index. Gram consumes nothing from radial, so it runs last. `disco.py`'s
`_radial_spectral_inputs` is the single place that assembles this; the packing order into the
logging buffers is deliberately *unchanged* (`upd, w, gram, gram_vec, radial, upd_spec, w_spec`),
since each family writes into its own preallocated slice and computation order is independent of
buffer layout.

**Spectral / radiality metrics.** `RADIAL_METRIC_NAMES` grew from 15 to 26. The new entries fall
into three groups:

- `spectral_radius`, `spectral_radius_next`, `spectral_growth`, `spectral_relative_step` --
  operator-norm counterparts to the Frobenius `radius` / `relative_step`.
- `radiality_rms_to_rms`, `radiality_rms_to_inf`, `radiality_l1_to_rms` -- how much of the update
  points along a *norming covector* of the weight under each induced operator norm. For rms->inf
  and l1->rms the norming covector concentrates on the weight's largest row/column, so these are
  row/column reductions; for rms->rms it is `u1 v1^T`, which needs the weight's leading singular
  *vectors*. The dimension factors (`sqrt(d_in/d_out)`, `sqrt(d_in)`, `1/sqrt(d_out)`) appear in
  both `N(W)` and `N(U)` and cancel, so the implemented forms carry no dimension scaling.
- `aus_frobenius`, `aus_rms_to_rms`, `aus_rms_to_inf`, `aus_l1_to_rms` -- the distance
  `N(W/N(W) - U/N(U))` under each norm. `aus_frobenius` is free: it is identically
  `sqrt(2 - 2*radial_cosine)`. The row/column variants expand to
  `sq_W[i]/a^2 + sq_U[i]/b^2 - 2*dot[i]/(a*b)` and reuse the reductions the corresponding
  radiality already computed, so no difference matrix is ever materialised.

**`sigma_max(W_before)` is not free, unlike the other two.** `calculate_norm` is called on `-lr*u`
and on `pseudo_w`, never on the pre-update weight, so the pre-update spectral quantities come from
`optimizers/power_iteration.py` -- warm-started across logging events from a cache
(`_power_iter_v_by_param_id`) that is deliberately *not* checkpointed, since it only exists on
whichever rank materialises that parameter's full weight. **That module is currently a stub
returning random values** (`power_iteration.IS_STUB`), and while it is, `disco.py` skips it
entirely rather than feeding meaningless numbers in: `spectral_radius`, `spectral_growth`,
`spectral_relative_step`, `radiality_rms_to_rms` and `aus_rms_to_rms` stay at the 0 sentinel. The
other six new metrics are exact and live now. One exception: a 1-D parameter stands for `diag(v)`,
whose largest singular value is exactly `max|v|`, so `spectral_radius` *is* real there.

**Fixed arity is load-bearing.** `disco.py` sizes four flat logging buffers from
`len(RADIAL_METRIC_NAMES)`. Every metric must therefore be present in every returned dict for
every input -- 1-D, 3-D, degenerate, or with no spectral inputs at all. Unavailable metrics take
the same deterministic `0` sentinel as the other degenerate cases rather than being omitted.

**`sigma_update` and weight decay.** radial's displacement is
`U = pseudo_w - W_before = -lr*u - wd*lr*W_before`, which equals `-lr*u` only when `wd == 0`. So
the update spectrum is reused as `sigma_update` only in that case; with weight decay on, reusing
it would divide by the norm of a different tensor, and `_radial_spectral_inputs` declines to.

Two things worth knowing if you're reading the formulas or extending this:
- `R2(t)` (from the "radial error" identity below) is exactly the same running sum as `raw_A2` --
  only one accumulator is kept, not two.
- `relative_step` is the same formula as `gram_helper.py`'s `U_relative_step_fro`. Not an accidental
  duplicate -- this one is unconditional, that one is gated behind `gram_level`.

The canonical `angle` (angle between consecutive weight directions `q_t`/`q_t+1`) is computed via
`atan2(a_t*tangent_fraction, r_t + a_t*radial_cosine)`, not `arccos(<q_t, q_t+1>)` -- `arccos`'s
derivative blows up near `cos=1`, so small angles (most training steps) lose precision in fp32;
`atan2` doesn't have that issue. Both formulas compute the exact same geometric quantity (verified via
the underlying 2D-trigonometry identity), so the `arccos` version is kept too, as `angle_from_cos`, purely
as an independent sanity check -- not fed into the accumulators.

**Relative-degeneracy floor on `radial_cosine`/`tangent_fraction`/`radial_ratio`:** these divide by
`a_t` or `a_t^2`, so once the actual update is numerically negligible relative to the weight's own
scale -- e.g. a near-zero-lr step at the tail of a decay schedule -- the division is noise divided by
noise, and that noise floor is genuinely different between DDP and FSDP (different collectives: DDP's
all-reduce vs FSDP's all-gather sum gradients in a different order, and floating-point summation isn't
associative). Observed in practice as those 3 metrics disagreeing hugely between a DDP run and an FSDP
run specifically at a near-zero-lr step, while `radial_first_order` (no division, `2*dot_wu`) only
differed slightly, and `angle`/`angle_from_cos` didn't disagree at all -- atan2's inputs stay
well-conditioned as `a_t -> 0` (numerator `-> 0`, denominator `-> r_t > 0`), so `angle` doesn't inherit
this instability the way a division by `a_t` does. Fixed by widening `valid_wu` from a bare `a_t > 0`
to `a_t > 1e-6 * r_t` (`radial_helper._REL_DEGENERACY_EPS`) -- a step below that relative threshold now
deterministically reports the same degenerate sentinel (`radial_cosine=0`, `tangent_fraction=1`,
`radial_ratio=0`) regardless of which parallelism strategy computed it, while a genuinely small-but-real
step (e.g. `relative_step ~ 1e-3`) is well above the threshold and reports real signal, unclamped.

`alpha_fit`/`tau_fit` (fitting the angle-decay power law `theta_t = C*(t+tau)^-alpha` from the logged
`(t, angle)` history) is a deliberately deferred, offline/analysis-time follow-up, not optimizer
state: unlike everything above (a genuine O(1)-per-call update), fitting this needs some bounded
history of past angles and periodic (not per-step) nonlinear refitting to stay cheap at scale, and
`tau` enters the fit nonlinearly, so there's no simple closed-form running update for it.

**Checkpoint compatibility:** resuming from a checkpoint saved *before* `radial_state` existed
crashes. TorchTitan's checkpoint load goes through `torch.distributed.checkpoint`'s default
`LoadPlanner`, which has `allow_partial_load=False`; since `radial_state` is always present in the
live optimizer's `state_dict()` skeleton (lazy-inited unconditionally in `_build_param_lists`), an old
checkpoint missing that key raises `RuntimeError: Missing key in checkpoint state_dict: ...` during
DCP's own planning phase -- before `DiSCO.load_state_dict` ever runs, so nothing on the optimizer side
can catch or work around it. A model-only load (optimizer state excluded entirely, e.g.
`--checkpoint.initial_load_in_hf` / `initial_load_model_only`) sidesteps this, at the cost of *all*
optimizer state (fresh momentum too, not just `radial_state`) -- there's no way to keep momentum while
dropping only `radial_state` short of relaxing `allow_partial_load` checkpoint-wide, which was
deliberately not done here since that would also silently paper over genuinely missing keys elsewhere.

**`DiSCO.load_state_dict` override** (`disco.py`) exists for two reasons, unrelated to the crash above:
1. `torch.optim.Optimizer.load_state_dict` overwrites every param_group key (besides `"params"`) with
   whatever the checkpoint saved, including config-derived keys (`eps`/`norm_factor`/`backend`/etc.,
   see `_CONFIG_ONLY_GROUP_KEYS`) that come from this run's config, not from training -- so resuming
   after a deliberate config change (e.g. switching `norm_factor`) would otherwise silently revert it.
   The override snapshots those keys before the base call and restores them after, warning on any
   mismatch.
2. `_momentum_buffer_by_param_id`/`_radial_state_by_param_id` (built once, normally at `__init__`) are
   refreshed afterward. This turned out to be defense-in-depth rather than a fix for an active bug: in
   TorchTitan's actual `dcp.load()` resume path, `OptimizersContainer.state_dict()` returns
   `self.state[p]`'s tensors *by reference* (`Optimizer.state_dict()` never clones), and DCP fills them
   in place before `load_state_dict` ever runs -- confirmed via an actual `dcp.save`/`dcp.load`
   round-trip, not just by reading source. So the caches stay valid on their own in that path; the
   refresh only matters for a load path that bypasses DCP's in-place fill (e.g. a direct/manual
   `load_state_dict()` call with a hand-built or detached state dict).

### Known limitation: `all_gather` sends to every rank, only one needs it

Every collective in this norm/gram/spectrum pipeline (`step_experts`'s single `all_gather_tensor`,
`step_ddp`/`step_fsdp`'s Phase-D `all_gather_tensor` via `_pack_segments`) gathers to **all** ranks,
but only the logging rank (`is_dp_rank_0` / FSDP-mesh rank 0) ever reads the result — every other
rank receives (and immediately discards) the full per-rank payload for nothing. Switching to a
root-only `dist.gather` would cut that wasted receive traffic without losing any fidelity (unlike
reducing the logged data itself, which isn't an option once you want the full vectors — see
`gram_vector_logging.py`'s docstring). Not done here: NCCL's support for plain `gather`-to-root is
limited/version-dependent, and this same inefficiency predates the gram-vector work (it already
applied to `track_spectrum_*`/scalar norms too) — fixing it is a genuine follow-up, not scoped into
either pass, and would need verifying against whatever backend/PyTorch version is actually in use
before landing.

---

# Pass 2 — per-rank logging correctness, batched radial, and a cost model

## Who logs what: one rule, one helper

`metrics.save_all_shard_ranks` means: **give a logger to every rank that owns a distinct slice
of the per-parameter metrics, and to no other.** Concretely — open the mesh DiSCO partitions
ownership over, and require local rank 0 in every other mesh. That is
`distributed/utils.rank_owns_metrics_shard`, and it is the single source of truth: both
`components/metrics._build_metric_logger` and `disco.py` call it, so "this rank stores metrics"
and "this rank can log" cannot drift apart.

Which mesh, mirroring `get_param_type`'s own branch:

| topology | ownership over | logger on |
|---|---|---|
| `fsdp_enabled` (every HSDP config) | `fsdp` mesh | any fsdp rank, `dp_replicate` 0, `tp` 0 |
| not `fsdp_enabled` (pure DDP) | `dp_replicate` mesh | **any `dp_replicate` rank**, `tp` 0 |

**The bug this fixed.** `step_ddp` partitions parameters over `dp_replicate`
(`_ddp_owner_rank_by_param = i % world_size`), and `get_param_type` returns `DDP` *only when
`fsdp_enabled` is False* — so DDP params exist exactly when there is no shard dimension to spread
logging over. The old predicate hard-coded "dp_replicate rank 0", so ranks 1..R-1 computed their
slice, stored it, and had no logger: **(R-1)/R of every per-parameter metric silently dropped** in
a pure-DDP run. HSDP was never affected (the `fsdp` mesh is orthogonal to `dp_replicate`).

`trainer.py` now also asserts the invariant per rank at init — *if this rank stores metrics it must
hold a logger* — so a future divergence fails loudly instead of producing a partial dashboard.

### Two store gates, not one

- `_stores_norms` — for the **rank-disjoint** families (fsdp, expert, ddp). Under per-rank logging
  it returns `rank_owns_metrics_shard`, not `True`: the dp_replicate replicas recompute identical
  metrics and have no logger, so without this 15 of every 16 ranks at `dp_shard=64` on 1024 GPUs
  would unpack and build a full metrics dict for a no-op `LoggerContainer`.
- `_stores_replicated_norms` — for **embed and scalar**, which are materialized whole on every rank
  and are therefore *not* rank-disjoint. Pinned to shard-local rank 0.

This was measured, not assumed: before the pin, the four ranks' key sets summed to 4,140 more than
the gathered path, and the excess was exactly 30 params × 46 metrics × 3 redundant copies — the
MoE router gates plus `tok_embeddings` and `output`, all bit-identical across ranks.

**Only local work may be skipped by either gate.** Every collective (`full_tensor()`, `lmo`, the
a2as, the gathers) runs on every rank or the job deadlocks, and `calculate_radial_metrics` runs
everywhere regardless because its accumulators live in `self.state[p]` and go through DCP — skip it
on some ranks and a resharded checkpoint disagrees with itself.

`verify_disco_ordering.py` asserts by AST which gate each `step_*` uses. Mixing them up is silent
in both directions (dropped shard metrics, or duplicated global ones), which is why it is a test.

## Batched radial metrics

`calculate_radial_metrics` takes `batch_ndim`. With `batch_ndim=1` and `W_before` of shape
`[E,m,n]`, every reduction is over the trailing matrix dims and every metric comes back as `[E]`.
The expert accumulators are already `(num_local_experts,)`, so they pass straight through.

It is one function with a batch axis rather than a second implementation, specifically so the 26
metrics and their degeneracy guards cannot drift apart. `verify_radial_batched.py` asserts both
directions: batched == looping the scalar path, **and** the scalar path still bit-identical to
`_radial_helper_pre_batch_reference.py`, a verbatim pre-refactor snapshot.

Two things worth knowing if you touch it:

- The unbatched `u1 @ (U @ v1)` is kept verbatim under `if batch_ndim == 0`. Both einsum and a
  singleton-padded matmul route to a different BLAS kernel and shift the last fp32 digit; a
  performance refactor must not move values that are already being logged.
- `linalg.vector_norm`, not `Tensor.norm(dim=...)`: the latter sends a 2-tuple of dims to
  `matrix_norm` and rejects anything else, so a 3-D tensor with `batch_ndim=0` would raise.
- `angle_from_cos` is the one metric that does **not** match the loop at fp32 tolerance. It is
  `arccos(cos)` at cos ≈ 0.99995, where the derivative is ~1e2, so a last-digit difference in the
  reduction order is amplified ~1000×. That ill-conditioning is exactly why the primary `angle`
  uses `atan2` — and `angle` does match. This is expected, not a bug.

## What a logging step actually costs (projection, not a measurement of a real run)

`tests/unit_tests/disco_metrics/analyze_logging_cost.py`. Method: build the flavor on the **meta**
device and enumerate its real parameters, route them through DiSCO's own rules, shard them per
family, then measure each distinct shape on one GPU with warmup and CUDA events and sum. It
independently reproduces the known "282 matrices of `[768,2048]` per rank" figure for
qwen30b-a3b @ EP=64, which is the cross-check that it is enumerating the right thing.

Per-rank compute per logging step, `norms_to_log=all`, GH200:

| flavor | dp_shard | norm | radial before | radial batched | total before → after |
|---|---|---|---|---|---|
| qwen30b-a3b | 64 | 587 ms | 472 ms | **54 ms** | 1059 → 642 ms |
| qwen30b-a3b | 128 | 383 ms | 280 ms | **44 ms** | 663 → 427 ms |
| mis-8b | 64 | 172 ms | 165 ms | **25 ms** | 337 → 197 ms |

So batching took radial from ~45% of the cost to ~8%, and **norm is now 91% of what remains.**

**This bounds per-rank compute only.** It says nothing about collective latency, contention, or the
full-weight all-to-all at 1024+ GPUs. Do not read the totals as a predicted step time.

### The lever that matters, measured

At the dominant expert shape `[768,2048] × 189`:

| norms_to_log | cost | vs all |
|---|---|---|
| `all` | 232 ms | 1.00× |
| `default` | 232 ms | 1.00× |
| sigma-only tier | 239 ms | 0.97× |
| no-SVD tier | **2.4 ms** | **95×** |

Two conclusions, both counter-intuitive enough to be worth recording:

1. **`default` costs exactly what `all` costs.** `condition_number` is in the default set and
   forces the full spectrum, so trimming `all` → `default` buys nothing at all.
2. **The sigma-only tier is not a speedup, and at the embedding shape it is a 3.4× pessimization**
   (594 ms vs 170 ms at `[201088,2048]`, where `gesvda` does badly). It is worth revisiting whether
   `SVDVALS_DRIVER_SIGMA_ONLY` should apply above some size.

The real lever is the spectrum itself: `condition_number`, `effective_rank*`, `stable_rank` and
`rms_to_rms` are the entire cost. Dropping them takes norm from 587 ms to roughly 10 ms. If those
metrics do not need to be at full `log_norm_freq` cadence, **logging them at a lower cadence than
the cheap norms is worth ~10× on logging compute** — a config knob, no new collective, and it
applies to every rank. That is a much better lever than the dp_replicate split below.

### dp_replicate logging split: analysed, not recommended

Under HSDP every replica computes bit-identical metrics — 16-fold redundant at `dp_shard=64` on
1024 GPUs. The redundancy is *parallel*, so today it costs no extra wall time; the prize would be
splitting the work so each replica does `1/R` of it, at the price of a new (small, scalars-only)
all_gather over `dp_replicate`.

At ~640 ms of per-rank logging compute for the 30B at `dp_shard=64`, and with `log_norm_freq=10`,
that is ~64 ms amortised per step. **Whether the split is worth it therefore depends entirely on
step time at scale, which this analysis cannot supply** — but the cadence change above is a larger
win with none of the structural risk, so do that first and re-measure before considering this.

## ns5 backends now agree

`newton_schulz_triton` used to special-case `steps == 5` to a tuned per-step schedule while
`zeropower_via_newtonschulz5` used a constant `(3.4445, -4.7750, 2.0315)` — same name, different
math, so switching backends silently changed training. The special case is gone; the Triton kernel
is now the reference iteration, just faster.

The cost is real and deliberate — orthogonality error at `steps=5`, `[768,2048]`:

| | error |
|---|---|
| old `newton_schulz_triton` (hidden schedule) | 0.036 |
| both, now | 0.310 |
| `polar_express_triton`, `steps=6` | **0.0036** |

If orthogonalization quality matters, use `polar_express_triton` — it is better than either ns5
variant and is the configured production backend. Do not reintroduce a hidden schedule.

## gram_helper: it was sync-bound, not compute-bound

`calculate_gram_metrics` at `gram_level=1` issued **951 CUDA kernel launches for 2.67 ms of
actual GPU time** — it was bound by CPU dispatch and by device→host syncs, not by linear algebra.
Two helpers caused the syncs:

- `_dominance_ratio` did `result[mask] = a[mask] / b[mask]`. Each masked access runs `nonzero()`,
  whose output shape is data-dependent, so each is a device→host sync — three per call, and the
  helper is called ~12× per `calculate_gram_metrics`. Now `torch.where`, which selects without a
  sync. The unselected branch may be inf/NaN and is discarded, the same deterministic-0 sentinel
  convention `radial_helper` uses.
- `_offdiag_stats` did `C[~torch.eye(m, dtype=bool)]` — an m×m bool mask plus a data-dependent
  gather of m²−m elements (589k at m=768) plus a sync. Now zeroes the diagonal in a copy.
  Deliberately **not** computed as `(total − diagonal)`: that is the catastrophic cancellation
  `_row_dominance` documents avoiding, and it bites exactly in the diagonally-dominant regime these
  metrics exist to detect.

Values verified identical (`verify_gram_helper.py`, including all-zero, diagonally-dominant,
single-row and negative inputs). **Interleaved** A/B at `[768,2048]` on an otherwise-idle GPU:

| level | boolean-mask indexing | torch.where / zeroed diagonal | |
|---|---|---|---|
| 1 | 6.0 ms | **4.5 ms** | 1.34x |
| 2 | 76.2 ms | 74.4 ms | 1.02x |

**A correction, and a lesson about this login node.** An earlier revision of this file claimed
~17x for level 1, from a non-interleaved A/B that measured the "before" arm at 78.8 ms with a
40-87 ms spread. That was **GPU contention from other users on the shared login node**, not the
change. The true effect is 1.34x. Any benchmark here must interleave its arms so contention hits
both equally, and a wide min-max spread is the tell that it did not.

The fixes are still right -- removing ~36 device-to-host syncs per call is correct regardless, and
syncs plausibly cost more under real training load than in an idle benchmark because they serialise
against queued work. But that is a hypothesis, not a measurement, and it does not rescue the 17x.

### Level 3: six sequential svdvals were a quarter of its cost

### Levels 2 and 3 are eigendecomposition-bound — batching is the lever

Profiling level 2: **14 separate `eigh`/`eigvalsh` calls on `[768,768]` account for ~165 ms of the
184 ms.** They are issued one at a time. Batched:

| | sequential | batched | |
|---|---|---|---|
| 14 × `[768,768]` eigh (values+vectors) | 261 ms | **96 ms** | 2.7× |
| 14 × eigvalsh (values only) | 226 ms | **65 ms** | 3.5× |
| 28 × eigh (2 local experts) | 524 ms | **146 ms** | 3.6× |

Batched float64 `eigh` is 110 ms — still **2.4× faster than today's sequential float32**, so the
accuracy upgrade below is available essentially for free once the calls are batched.

Not yet done: it needs `_level2_metrics` restructured to collect every matrix needing a
decomposition, issue one batched call, then scatter the results back. Mechanical, but invasive.

### A correction worth recording

The SVD fallbacks `_gram_eigh_from_factor` / `_svd_eigenvalues` are reached **only** when the
primary `eigh` raises `_LinAlgError` — instrumented, they fire **0 times** across levels 1–3 on
normal input. They were switched to `norm_helper.fp64_gram` (5.9× faster for values, 4.4× with
vectors, and 34–500× more accurate on the smallest eigenvalue), which makes the fallback better and
directly answers the concern their old docstrings raised — forming `X Xᵀ` squares the condition
number, which is fatal in float32 and affordable in float64. But it is **not** a speedup of the
normal path, and an earlier claim here that level 2 improved 2.47× from it was wrong: that number
came from one fast trial of a bimodal distribution (`[82–186] ms`). Re-measured with more trials,
level 2 was unchanged by that edit.

Note the *primary* path still forms the Gram in float32 (`_gram(X) = X @ X.T`) and eighs it, so it
carries the squared-condition-number accuracy loss the fallback was written to avoid. Moving the
primary path to `fp64_gram` is the accuracy half of the batching work above.

## spectrum_logging: batched device→host transfer

`process_norms_for_logging` did one `.cpu()` per spectrum tensor — ~672 per rank per logging step
(update + weight per tracked matrix), each its own sync and copy. `gram_vector_logging.py` had
already solved exactly this for its own vectors; the two files had simply diverged. Now one
`cat` + one `.cpu()` + host-side slicing: **10.0 ms → 5.3 ms**, values bit-identical. Spectra have
different lengths, hence explicit offsets; each slice is `.clone()`d for the same reason its
sibling clones — torch pickles the whole underlying storage a view points into.

### Gram: the numbers that survive interleaved measurement

Per matrix at `[768,2048]`, and per rank per logging step at `dp_shard=64` (337 matrices):

| level | per matrix | per rank | what changed |
|---|---|---|---|
| 1 | **4.27 ms** | 1.4 s | sync fixes, 1.34x |
| 2 | **73.8 ms** | 24.9 s | ~unchanged (1.02x) |
| 3 | **125.3 ms** | 42.2 s | batched fp64 Gram for the six svdvals, **2.40x** |

Level 3's win, verified by a 12-pair interleaved A/B: `_level3_metrics` ended with **six separate
`torch.linalg.svdvals` calls on `[m, m]` matrices** (`C_WV/C_WU/C_VA` and `Q_WV/Q_WU/Q_VA`),
301.6 ms of its cost. They all have the same shape, so they now go through one batched
`_gram_spectrum`: **301.6 -> 125.7 ms overall, saving 176 ms per matrix.**

Two things worth keeping from that measurement:

- **Batching `svdvals` itself buys nothing** -- measured 1.00x, because cuSOLVER serialises it
  internally. That is the same finding as pass 1's expert-norm batching. The win comes from the
  float64 Gram, which turns each SVD into an `eigvalsh` of an `[m, m]` matrix, and *that* batches.
- It is also **6800x more accurate**: max relative error 5.9e-8 against a float64 SVD reference,
  versus 4.05e-4 for the float32 `svdvals` it replaces.

Level 2 remains eigendecomposition-bound (14 sequential `eigh`/`eigvalsh` on `[768,768]`), and the
same batching trick applies there -- measured 2.7-3.6x available, not yet built.
