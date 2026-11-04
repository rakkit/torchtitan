# DiSCO

`DiSCO` (`disco/optimizer.py`) is a distributed Muon / Scion optimizer. Each
step it updates the momentum, turns each weight matrix's momentum into an update
direction with a per-parameter-group **rule** (an orthogonaliser, a row
normalisation, a Gram polar factor, ...), and applies

```
w <- (1 - lr * wd) * w - lr * scale(m, n) * U
```

`DiSCOOptimizersContainer` (`container.py`) builds one DiSCO per model part
from the config, runs the MoE load-balance hook, and hands the metrics to the
trainer. DiSCO computes every rule where the unit it needs is whole, with one
packed collective per kind of traffic, whatever the parallelism.

Momentum convention: `buf <- (1 - momentum) * buf + momentum * g`. `momentum`
is the weight of the NEW gradient, the opposite of Muon's: the production
recipes use 0.1 (90% memory); the constructor default 0.95 keeps 5%.
`nesterov: true` hands the rule `buf + momentum * (g - buf)` instead of `buf`.
A momentum outside (0, 1) bypasses the buffer: the rule gets the raw gradient.

## Package layout

```
optimizers/
  container.py      DiSCOOptimizersContainer: config, param groups, arming, metrics hand-off
  moe_balance.py    MoE load-balance bias hook; router metrics on logging steps
  model_stats.py    residual, RMSNorm-gain and sink statistics (log_*_stats)
  gnoise.py         gradient-noise logging; gnoise_research.py, gnoise_math.py
  utils.py          FQN cleanup, the DiSCO kwargs the container passes
  disco/
    optimizer.py    DiSCO: build (families, exchanges, metric registry), step(), state
    layout.py       families (classify), per-rank row tables
    rules.py        what a rule is: Phase, Compute, Chain, Rule, Result; the registry
    presets.py      the built-ins: pre phases, computes, presets sgd/muon/right_polar/
                    lm_head/left_polar/aus/hyperball/elr/sso/amphione/onerate/
                    angular_muown/lie_relative
    transport.py    Exchange (one packed all-to-all pair), Replicated (all-gather)
    metrics.py      MetricSink: per-unit registry, one flat buffer, one gather, unpack
    gram_input_stats.py  sigma_1 / ||X||_F of gram_polar_express's inputs (log_gram_input)
  kernels/          orthogonaliser backends: zeropower, triton Newton-Schulz,
                    Gram Newton-Schulz, underflow floors
  helpers/          norm_helper, gram_helper, radial_helper, head_geometry, power_iteration
  export/           spectrum / gram-vector Parquet export (W&B artifacts)
models/common/param_layout.py   the declaration vocabulary models use (ParamLayout, UnitSpec)
components/optimizer/schedule_tables.py   knots for group lr_table / eta table schedules
components/optimizer/averages.py          weight power averages, snapshots, scoring (averages)
observability/nanprobe.py                  env-gated NaN probe (debugging)
```

## Configuring it

A run states named parameter groups; the production recipe:

```python
config.optimizer.groups = {
    "default": {"lr": 0.1, "weight_decay": 0.0, "momentum": 0.1, "nesterov": False,
                "rule": "muon",
                "args": {"compute": "gram_polar_express", "steps": 6,
                         "scale": "spectral", "pre": "identity"}},
    "embedding": {"match": r"tok_embeddings\.weight",
                  "args": {"compute": "row-l2", "scale": "sqrt_n"}},
    "lm_head":   {"match": r"(output|lm_head)\.weight",
                  "args": {"compute": "row-l2", "scale": "inv_sqrt_n"}},
    "router":    {"match": r"\.router\.gate\.weight",
                  "args": {"compute": "row-l2", "scale": "inv_sqrt_n"}},
}
```

- `default` is required; every other group inherits from it and selects its
  parameters with exactly one of `match` (an FQN regex) or `kind` (a declared
  kind, e.g. `{"kind": "unembedding"}`). Groups are tried in order, first match
  wins, the rest go to `default`.
- Group keys: `optimizer` (default `"DiSCO"`; `"AdamW"` sends the group to an
  AdamW beside DiSCO), `lr`, `weight_decay`, `decoupled_wd`, `momentum`,
  `nesterov`, `rule`, `args`. Every key the config file sets is a CLI flag:
  `--optimizer.groups.default.lr 0.05`, `--optimizer.groups.default.args.steps 8`.
- Weight decay scales W by `1 - lr weight_decay` per step, or by
  `1 - weight_decay` with `decoupled_wd: true`, independent of the lr and its
  schedule. The rule's own step is unaffected. `weight_decay` and
  `decoupled_wd` both come from the config, never the checkpoint, so a
  resume cannot pair one run's decay with another's mode.
- `args` merge over the default group's. A group with the default's `rule`
  takes all of the default's args; a group with another rule takes only the
  ones its preset accepts, minus the preset's `own` ones (below). So a group
  whose rule differs must state every arg that should not come from the
  default: under a `muon` default with `compute: gram_polar_express`, a bare
  `{"rule": "sgd"}` group inherits that orthogonaliser.
- Build errors: unknown keys; the renamed ones (`backend`, `norm_factor`,
  `pre_norm`, which name their replacement); DiSCO-only keys (`rule`, `args`,
  `momentum`, `nesterov`, `decoupled_wd`) on another optimizer's group; a
  DiSCO with no default `lr`; `weight_decay` in a `hyperball` or `sso` group;
  `gram_level` outside 0-3; an iterative compute with `steps < 1`; the
  vocabulary matrices under a gathering rule; a Gram compute on a wide unit;
  declared units that overlap or misstate their shape; a stateful rule whose
  state shape differs across one parameter's units.
- `moe_template_config()`'s groups are a valid run on their own: the production
  recipe (amphione on the matrices with the router gate's row lengths frozen, a
  sign step on the RMSNorm gains, row-l2 on the embedding and the head).
- DiSCO groups are written as `optimizer.groups`; upstream's
  `optimizer.param_groups` is for other optimizers only (DiSCO in it is an
  error: only `groups` marks the named groups apart from the default one), and
  setting both is an error.

Container fields (`--optimizer.<field>`):

| field | default | |
|---|---|---|
| `groups` | `{}` | the named parameter groups above |
| `communication_dtype` | `float32` | dtype of the momentum and updates that travel; `bfloat16` selectable, `float16` refused |
| `log_norm_freq` | 0 | per-parameter metrics every N steps (and at step 1 and the last step); 0 = off |
| `norms_to_log` | `["default"]` | norm names, `"default"` or `"all"` (see Metrics) |
| `effective_rank_freq` | 0 | N > 0: `effective_rank` (the whole spectrum) only on the norm-logging steps divisible by N, step 1 and the last step; 0 = every norm-logging step |
| `spectral_norm_freq` | 0 | N > 0 (a multiple of `log_norm_freq`): every SVD-based metric (`rms_to_rms`, `stable_rank`, `condition_number`, `effective_rank*`, the spectra, the six spectral radial metrics) only on the norm-logging steps divisible by N, step 1 and the last step; the SVD-free norms and the other radial metrics keep `log_norm_freq` (dense phase included). 0 = every norm-logging step |
| `gram_level` | 0 | Gram metrics, 0 = off, 1-3 cumulative |
| `head_diag_probes`, `head_diag_diameter_iters` | 512, 10 | LM-head geometry: tokens of the last microbatch whose head input it uses (0 = off), diameter iterations |
| `enable_spectrum_export`, `enable_gram_export` | False | Parquet export of the singular-value spectra / Gram vectors, one `step_N_rankR.parquet` per writing rank |
| `enable_step_tracing` | False | structured-log records per step |

Why float32 transport: only the sharded matrices travel (attention and shared
experts, ~7% of mis-8b's parameters; under pure DDP also the replicated units'
updates). With bf16 the production
`gram_polar_express` receives 8 mantissa bits where it works in fp16's 11;
measured on real step-500 momenta, the update direction moves ~3% (up to 16% on
deep attention). The cost: +3-4.5 ms per optimizer step at dp_shard 8-16 on
mis-8b, +0.05 GB/GPU of buffers at dp_shard 64.

## What the model declares

DiSCO never infers what a parameter is from its name or shape. The model
declares every parameter (`param_layouts() -> {fqn: ParamLayout}`, types in
`models/common/param_layout.py`, so model code does not import the optimizer):

| kind | what it is |
|---|---|
| `matrix` | a weight matrix; a fused projection lists its units (q/k/v, w1/w3) as row runs |
| `expert_batch` | dim 0 is a batch of experts: one unit each, or the declared units of each (w1, w3 of a fused `[E, 2, F, D]` w13) |
| `embedding`, `unembedding` | the vocabulary matrices, rows are tokens |
| `router` | an MoE router's gate, rows are experts |
| `conv` | a depthwise Conv1d `[C, 1, K]` (the attention's short conv): one `[C, K]` unit, rows are channels |
| `vector`, `scalar` | a one-row unit `[1, d]` / `[1, 1]` |

opt_moe's projections are fused, as upstream's (`wqkv`, `w13`, and the
experts' `w13` `[E, 2, F, D]`). Its modules declare what only they know (expert
batches, the q/k/v and w1/w3 units, the router gate), finding each parameter by
identity (`named`); the
model adds the two vocabulary matrices and states the rest by rank (`by_rank`,
which refuses rank 3+: a depthwise Conv1d `[C, 1, K]` and an expert batch
`[E, F, D]` look alike). A fused parameter's units are unions of contiguous
row runs, so they get unfused math and log under the split names
(`...wqkv` + `wq` logs as `...wq`; an expert's `ep_3/...w1`).

Checked at build: an undeclared parameter; a vocabulary matrix under a rule
that would gather all its rows on one rank; a Gram compute on a wide unit
(rank-deficient Gram); a unit a compute cannot take (`check_unit`: head-wise
blocks that do not divide it); and the ranks of a model part compare a hash of their
plans, so a divergent plan raises instead of hanging the first exchange.

## Rules: pre -> compute -> post -> scale

A rule is a small optimizer over one unit, run as a pipeline; the optimizer
derives every collective from what its pieces declare, and a rule never sees a
transport.

| stage | config (`args`) | choices |
|---|---|---|
| pre | `pre` | `identity`, `row-l2`, `col-l2`, `mat-l2`, `col-center`. A phase acts on the rows a rank holds, or declares `reduce` and supplies a fixed-size statistic summed over every rank holding rows of the unit (one packed all-reduce for all units). Phases are told the unit's true row count, never a shard's |
| compute | `compute` (+ `steps`) | the only place a direction is formed. Row maps `identity`, `row-l2`, `sign`, `row-l2-deflate` (row-l2 after removing the momentum's top right singular direction from every row; needs the whole unit); orthogonalisers `newtonschulz5`, `newtonschulz5_triton`, `polar_express`, `polar_express_triton`, `gram_polar_express`, `svd`; `gram_polar` (the polar factor through the `[n, n]` Gram); chains `main+row_map+...` (`gram_polar+row-l2`) |
| post | (preset-internal) | phases that need a statistic over all rows after the compute (the LM head's re-centring) |
| scale | `scale` | the unit's step relative to `lr`, from its shape `[m, n]` = `[d_out, d_in]`: `one`, `spectral` (sqrt(m/n)), `image_spectral`, `n`, `sqrt_n`, `inv_n`, `inv_sqrt_n`, `sqrt_mn`, `rms_parity` (min(sqrt(m/n), 1): unit rows at the Frobenius norm of polar x spectral). A constant per unit, the last multiply of the compiled compute (for a Gram compute, or after a reducing post phase, it runs after it) |

A `Compute` declares where the unit must be whole (`kind`: `rows` = the rows a
rank holds, `gram` = those rows plus one `[n, n]` all-reduce, `unit` = whole on
its owner), whether it needs the pre-update weight (`needs_weight`), whether it
keeps state (`has_state`, shape `state_shape(m, n)`) and what it returns
(`output`). One apply serves every output: `"update"` ->
`w <- decay w - lr U`; `"update_coeffs"` -> per unit `w <- alpha w - U`, the
step's size folded into U (alpha rides one small all-reduce); `"weight"` ->
`w <- W_next` (the step `W - W_next` travels in the update's slot). The apply
adds lr and weight decay only to a plain update: a coefficient or weight
output must fold them in itself, from the `ctx` it is given
(`StepCtx(lr, wd, t, decoupled)`, whose `decay` is `1 - lr wd` or `1 - wd`),
and cannot have post phases or a scale. A contract that cannot hold fails
when the rule is built, not at the first step.

Presets, each a function registered with `@register(name, own=...)` whose
keyword arguments are its args and defaults:

| `rule` | pipeline | defaults | `own` |
|---|---|---|---|
| `sgd` | `pre` -> `compute` -> scale | identity, identity, steps 0, `one` | |
| `muon` | `pre` -> orthogonaliser -> scale; per block with `block_rows` / `block_cols` | identity, `newtonschulz5`, 5, `spectral`, blocks 0 | |
| `right_polar` | `pre` -> `gram_polar` -> scale; no all-to-all | 5 steps, `spectral`, work dtype fp16 | `compute` |
| `lm_head` | centre -> `gram_polar+row-l2` (`spectral: false`: `row-l2`) -> centre -> scale | 5 steps, `inv_sqrt_n` | `scale` |
| `left_polar` | centre -> orthogonaliser -> centre -> scale (routers) | `newtonschulz5`, 5, `spectral` | |
| `aus` | `AUSMap` around a muon direction: needs the weight, keeps a per-unit vector, returns `beta U` and `alpha` (beta folded into the update) | as `muon`, `sigma_iters` 8, `eta` none | |
| `hyperball` | `FrobeniusAUS`, radius kept: `W+ = r N(W - s r N(U))`, r = `\|\|W\|\|_F` | as `muon` without a scale; `tangent` false, `exact_angle` false, `eta` none | `scale` |
| `elr` | `FrobeniusAUS`, radius free: `W+ = d W - s r N(U)`, d the decay | as `hyperball` | `scale` |
| `sso` | `SpectralSphere`: `W+ = (R / sigma) W - s R orth(M^ + lambda* u v^T)`, `R = scale(m, n)` | as `muon`; `sigma_iters` 8, `solve_steps` 12, `lambda0` 1e-3, `eta` none | |
| `amphione` | `AmphioneMap`: `W = Diag(g) U`, unit rows turned by the orthogonalised momentum, row lengths by a sign step, one rate: the group's lr from the trainer's scheduler; optional column gains (see "Amphione" below) | `gram_polar_express` 6, `beta` 0.95, Nesterov, `gain_beta` = `beta`, columns off | |
| `onerate` | `angular_muown` with the optimizer study's Amphione settings as defaults (radius 4, row-length average 0.9), every switch open: for ablations | as the study ran | `scale` |
| `angular_muown` | `AngularMuownMap` (arXiv 2606.23637) and the study's switches: length rules, sphere, tangent, radius, turn norm, frozen units, column scales | `polar_express` 5, `image_spectral`, warmup 100, Adam lengths | `scale` |
| `lie_relative` | `LieRelativeMap`: a relative step in the RMS -> RMS norm, `\|\|dW P^-1\|\|_S = eta` (`P = sym(Q^T W)`, `Q = Orth(W)`), second-order exp map, transported momentum | `gram_polar_express` 6, `one`, `beta` 0.95, Nesterov | `scale` |

`right_polar`, `lm_head` and `left_polar` are the symmetry-compatible rules of
Lau & Su (arXiv 2605.18106); the paper's other two are configs: row-norm only
is `{"rule": "sgd", "args": {"pre": "row-l2", "compute": "identity", "scale":
"one"}}` (stated in full, since an `sgd` group under a `muon` default inherits
its compute and scale), the SwiGLU hybrid `{"rule": "muon", "args": {"pre":
"row-l2"}}`. `aus` over a row compute stays that row rule; over a Gram compute
it is a build error. The production row rules are
`compute: row-l2` with `scale: sqrt_n` (embedding) or `inv_sqrt_n` (head,
routers). `lm_head` and `left_polar` take no `pre`: the centring is the rule.

**Head-wise Muon.** `muon` with `block_rows: head_dim` orthogonalises each head
of a q, k or v unit on its own, `block_cols: head_dim` each head's columns of
`wo`; each block is scaled by its own shape (`spectral`: sqrt(rows / cols) of
the block). The block size must divide every unit of the group (checked at
build). Head-wise attention is two groups:

```python
"qkv": {"match": r"\.wqkv\.", "args": {"block_rows": 128}},
"wo": {"match": r"attention\.wo\.", "args": {"block_cols": 128}},
```

**The attention's short conv and sinks** (opt_moe `conv_kernel_size`,
`attention_sinks`). A conv weight `[C, 1, K]` is one `[C, K]` unit (kind
`conv`, rows are channels); the sinks are one `[1, n_heads]` row (kind
`vector`). Select them with groups:

```python
"conv": {"kind": "conv", "args": {"compute": "row-l2", "scale": "one"}},        # row-norm
"conv": {"kind": "conv", "args": {"compute": "polar_express", "steps": 5}},     # orthogonal
"sinks": {"match": r"\.sinks$", "args": {"compute": "sign", "scale": "one"}},
```

- **Row-norm** normalises each channel's K taps. It runs on the channels a
  rank holds, so it needs no communication.
- **Orthogonal** is muon on `[C, K]`, gathered to an owner. With K = 4 the
  tiled orthogonalisers cannot run: `gram_polar_express` needs dims divisible
  by 8, and the Triton ones need at least 16. They are refused at build, so use
  `polar_express`, `newtonschulz5` or `svd`.
- **Sinks:** under the production default, the `[1, n_heads]` unit is refused
  too (one row). Give them a group: `sign` moves each head's sink by lr;
  `row-l2` normalises the whole vector; muon on a pure-PyTorch backend is the
  `[1, H]` polar, i.e. `row-l2` times `sqrt(1/H)`. None was clearly better
  (below).
- **Not yet trained with:** `right_polar` (the polar factor through the
  `[K, K]` Gram) orthogonalises `[C, 4]` exactly at 8 steps, with a 4x4
  all-reduce and no gather. NS5 and polar express leave singular values in
  0.68-1.10.
- **Measured** (mis-8b-proxy, 300 steps, one seed; train loss over steps
  251-300):

  | setting | train loss |
  |---|---|
  | no conv / sinks | 6.596 |
  | row-norm conv + sign sinks, default lr 0.1 (recommended) | 6.556 |
  | row-norm conv + sign sinks, lr 0.01 | 6.570 |
  | orthogonal conv + sign sinks, lr 0.1 | 6.632 |
  | orthogonal conv + sign sinks, lr 0.01 | 6.644 |

  Sink rule, with row-norm conv at lr 0.1 (H = 8; train over steps 251-300 /
  val at 300):

  | sink rule | step per head (RMS) | train | val |
  |---|---|---|---|
  | `sign`, `one` (two runs) | lr | 6.555 / 6.569 | 6.774 / 6.771 |
  | `row-l2`, `one` | lr / sqrt(H) | 6.578 | 6.779 |
  | `row-l2`, `sqrt_n` | lr | 6.577 | 6.772 |
  | muon, `polar_express`, `spectral` | lr / H | 6.555 | 6.764 |

  The runs are not deterministic: the same `sign` config twice differs by
  0.014 (train) and 0.003 (val). The four rules span 0.023 and 0.015, so no
  sink rule stands out at this scale; all beat no conv / sinks.

**Sphere rules.** `hyperball` (MuonH, arXiv 2606.16899), `elr` (Tencent's ELR,
hy.tencent.ai/research/elr, their Eq. 2) and `sso` (arXiv 2601.08393) need the
weight on the owner and return per-unit coefficients, like `aus`. Their step
size s is the group's `lr` (the papers' setting), times `eta(t)` when `eta`
is an `EtaSchedule` dict (see "Angular schedule" below).
- `hyperball` and `elr` are AUS in the Frobenius geometry: s is the ELR, the
  step relative to `||W||_F`.
  - `exact_angle: true` solves s in closed form so that the angle between W
    and W+ is exactly s (arXiv 2607.22444).
  - `tangent: true` (MuonH-T) removes the momentum's component along W before
    the orthogonaliser.
  - `N()` cancels any scale, so their `scale` applies only to a row compute:
    an embedding group that inherits the rule keeps its own.
- `sso` is steepest descent on the spectral sphere of radius `scale(m, n)`,
  which is `sqrt(d_out / d_in)` by default; the spectral angle of its step is
  about s.
  - It needs the top singular pair of W (a warm power iteration) and the root
    of `h(lambda) = <u v^T, orth(M^ + lambda u v^T)>`.
  - The root is bracketed by steps growing 4x from `lambda0`, then found by
    Illinois false position (a secant that keeps the bracket), in
    `solve_steps` orthogonaliser calls after the one at 0. The count is the
    same for every unit, so there is no host sync.
  - The default 12 resolve |lambda*| up to 0.57 to 2e-5. Plain bisection
    after doubling, the paper's scheme, reached only 6e-2 there. Real hidden
    matrices measured <= 0.03.
  - It costs `1 + solve_steps` orthogonalisations per unit, against Muon's 1.
- `hyperball` and `sso` hold each unit's norm, which undoes weight decay
  exactly, so their groups must set `weight_decay: 0` (checked at build).
  `elr` keeps weight decay.
- A radius-relative step never moves a zero-norm unit under `hyperball` or
  `elr`; the first step warns if one exists.
- `sso` takes `unit_eta`, `{unit type: factor}` on each unit's step (unit
  types as DiSCO logs them without `layers.<i>.`).
- Like `aus`, they are meant for the hidden matrices. A row group (embedding,
  head, routers) that inherits a sphere rule stays that row rule, unless
  `row_direction: true` (`hyperball`, `elr`, `angular_muown`, `onerate`,
  `amphione`) makes the row compute (`row-l2`) the sphere rule's direction.
  - The default group's `lr` is the ELR, and a row group without its own
    `lr` inherits it. `moe_template_config()`'s row groups set none, so give
    them one.
  - The ELR post's minus-square-root schedule is the trainer's
    `lr_scheduler` with `decay_type: "sqrt"` and `decay_ratio: 1.0`, or an
    `eta` of shape `sqrt_decay` on top of a WSD.

**Angular schedule.** Every matrix rule but `amphione` (`muon`, `aus`, `hyperball`, `elr`,
`sso`, `angular_muown`, `onerate`, `lie_relative`) takes `eta`, an `EtaSchedule` dict that multiplies
the group's lr for the direction step only: the step is `lr_t eta(t)`, lr_t
the trainer's schedule. Weight decay keeps `1 - lr_t wd`, and angular-muown's
magnitudes keep lr_t. `eta(t)` is 1 up to `start`; after it, with
`x = t - start`, `inverse_poly` is `(1 + c x)^-p` (angular-muown's kappa) and
`sqrt_decay` is `1 - sqrt(x / (total - start))` (the ELR post's msqrt). A
shape takes only its own keys; `table` takes `points`, `((step, value), ...)` with increasing steps: eta is linear between them and the first / last value outside. A row group under a default with `eta` keeps
its lr (`muon` does not pass `eta` on; the sphere rules return the row rule).
The value used is logged as `eta_schedule/DiSCO/<group>` and
`eta_used/DiSCO/<group>` (lr_t eta(t)); `lr/DiSCO/<group>` is lr_t alone, so
with an `eta` the applied step is `eta_used`, not `lr/`. A schedule can live
in the trainer instead: `lr_scheduler.decay_type: "sqrt"` with
`warmup_steps: 0` and `decay_ratio: 1.0` is `1 - sqrt(t / T)` at the
optimizer's step t, the same curve as `eta` of shape `sqrt_decay` with
`start: 0`, applied to every group, and `lr/` is then the applied rate.

**Per-group schedules.** A group may replace the trainer's schedule by its
own keys: `warmup_steps`, `decay_ratio`, `decay_type`, `min_lr_factor`, or
`lr_table`, knots `((step, factor), ...)` on the optimizer's step t (linear
between them, the first / last factor outside; not with the other keys).
`components/optimizer/schedule_tables.py` builds knots for the trainer's WSD,
a horizon-free base `(1 + D / D1)^-q` in tokens and a cooldown of any table.
Checkpoints leave `lr_table` out, so a run resumed from another run's
checkpoint takes its own.

**Amphione** (`rule: amphione`). Each hidden matrix is `W = Diag(g) U`: U has
unit rows, g holds their signed lengths. From the raw gradient G, with
`eta_t = lr` (the group's lr at this step, from the trainer's scheduler; the
rule takes no `eta` schedule) and `b = gain_beta` (None: `beta`):

```
gamma = rowdot(G, U),  D = Diag(g) G
M <- beta M + D,  O = Orth(D + beta M)      (Nesterov; M without)
O <- O ||U||_F / ||O||_F
U <- RowNorm(U - eta_t O)
m_g <- b m_g + (1 - b) gamma,  g <- g - eta_t sign(m_g)
```

g starts at the step-1 row norms (a zero row at sqrt(0.33)); there is no
step-1 rescale, so a larger start belongs in the init. Args:

| arg | default | meaning |
|---|---|---|
| `compute`, `steps` | `gram_polar_express`, 6 | the direction: an orthogonaliser (`polar_express`, `newtonschulz5`, `svd`, ...), or `row-l2` with `row_direction` (below); a Gram compute is refused |
| `row_direction` | false | with a row compute: true = Amphione with that row compute as the direction; false = the plain row rule (below) |
| `scale` | `one` | used only by that plain row rule (Amphione rescales O to `\|\|U\|\|_F`); never inherited from the default group |
| `beta`, `nesterov` | 0.95, true | the direction's momentum |
| `gain_beta` | None (= `beta`) | the average the row-length (and column) sign steps take |
| `columns` | false | column gains c, `W = Diag(g) U Diag(c)`, on `column_units` |
| `column_units` | wq, wk, wv, w1, w3 | the unit types that learn c (`AMPHIONE_COLUMN_UNITS`) |
| `column_lr_mult` | 0.0625 | c's sign step is `column_lr_mult * eta_t` |
| `frozen_length_units` | () | unit types of the group whose row lengths g stay at their start; their rows still turn (the MoE router gate: `["moe.router.gate"]`, the study's MoE setting, there `onerate` / `angular_muown` `frozen_units`); each type must name a unit of the group; never inherited from the default group; not with `columns` |

**Row computes and `row_direction`** (as on `hyperball`, `elr`,
`angular_muown`, `onerate`):

| `compute` | `row_direction` | the rule |
|---|---|---|
| an orthogonaliser | false | Amphione, `O = Orth(D + beta M)`: the hidden matrices |
| `row-l2` | **true** | Amphione, `O = rowl2(D + beta M)`: the orthogonaliser is replaced by normalising each row on its own; everything else (unit rows, the rescale, the row lengths' sign step, the rate) is the same |
| `row-l2` | false (default) | **not Amphione**: the plain row rule `W <- W - lr scale(m, n) rowl2(M)` |
| an orthogonaliser | true | an error |

The fall-back exists because a group without its own `rule` inherits the
default group's. An embedding group written `{"args": {"compute": "row-l2",
"scale": "sqrt_n"}}` under an `amphione` default stays a plain row-norm step,
not Amphione with learned row lengths. To be unambiguous, give every group
its own `rule` (the recipe does: `sgd` for the RMSNorm gains, `muon` for the
embedding and head).

`row-l2` as the direction is for the attention's short conv, a `[C, K]`
unit (K = 4): every channel's filter is its own `1 x K` matrix, whose
spectral normalisation is its l2 norm (the tiled orthogonalisers cannot run
on K = 4 anyway). The optimizer study's conv group, at the hidden matrices'
lr:

```python
"conv": {"match": r"(^|\.)attention\.(q|k|v)_conv\.weight$", "rule": "amphione",
         "lr": 0.01562, "momentum": 1.0, "nesterov": False, "weight_decay": 0.0,
         "args": {"compute": "row-l2", "row_direction": True}},
```

It inherits the default group's other `amphione` args (`beta`,
`gain_beta`, ...). With `columns: true` on the default, set `columns: false`
on the conv group: the conv units are not in `column_units`, and DiSCO
refuses a column unit type that matches no unit of the group. The study ran
this group with the row lengths x4 at step 1 (`radius: 4`) from
`conv_init_gain: 1`; without a step-1 rescale the counterpart is
`conv_init_gain: 4` (not run).

The rule keeps its own momentum: its group sets `momentum: 1.0`,
`nesterov: false`, `weight_decay: 0`. Its state, `[M | g | m_g | r]` per unit
(three more rows `[c; m_c; r]` with columns, `r` unused), has the layout the
optimizer study's runs checkpointed. A group config:

```python
"default": {"rule": "amphione", "lr": 0.01562, "momentum": 1.0, "nesterov": False,
            "weight_decay": 0.0,
            "args": {"compute": "gram_polar_express", "steps": 6,
                     "polar_express_safety": 1.05}},
```

What the optimizer study ran, where it differs from these defaults: Gram
Polar Express at `polar_express_safety` 1.05; a constant trainer lr with
an `eta` table 1 - sqrt(t / T) (on `onerate`; `amphione` takes the trainer's
sqrt schedule instead); the row-length average 0.9 (angular_muown's default
`betas`, never run at 0.95); the row lengths scaled x4 at step 1 by the
optimizer (`onerate` `radius: 4`), measured to match x4 in the init within
0.003 in loss. `onerate` is that rule with every `angular_muown` switch
open, for ablations; with `radius: 1` and the same `gain_beta` it steps
bitwise as `amphione` (`tests/unit_tests/cpu/optimizers/test_amphione.py`).

**Column scales (`amphione` `columns: true`, `angular_muown` `col_scale: true`;
off by default).** Under `amphione` c takes the sign step above at
`column_lr_mult * eta_t` on the `column_units`. Under `angular_muown`: W =
Diag(g) U Diag(c) (MuonMD, arXiv 2606.25971 Eq. 1): c starts at 1 and is moved
by the row scales' rule (Adam or sign, same betas, eps and rate lr_t x
`magnitude_lr`) on dL/dc_j = sum_i G_ij g_i U_ij; U's gradient is Diag(g) G
Diag(c) and the row gradient rowdot(G Diag(c), U). c lives in three extra rows
of the rule state ([c; m_c; v_c], state (m + 3, n + 3)), so it is checkpointed
with it and sharded as the rule state (whole units on their owner). Needs
`magnitude_shape: row`, `magnitude_rule: adam | sign` and no `frozen_units`.
The parameter is the effective W, so every logged norm and radial metric is
W's; U, g and c are not logged.
- **Measured** on 1 node, 200 steps, mis-8b-proxy (MoE) and
  M1-dense-600M-full_rope-proxy, with the step size set on the hidden
  matrices, as the median of the step's angle / that size over the hidden
  units:
  - `hyperball` and `elr` hold the Frobenius angle at 0.98-1.00, and with
    `exact_angle` at 1.0000 on every unit and step.
  - `sso` and `aus` aim at the spectral angle. It starts at ~1.6x at step 1
    and falls to 1.03-1.08x (`sso`) or 1.05-1.14x (`aus`) by step 200.
  - That early excess is the weights, not the rule. An orthogonal init has a
    flat spectrum, where the spectral norm moves at first order in every
    direction: a synthetic step at a flat spectrum gives 1.8x, with a 1%
    top gap 1.2x, with a decaying spectrum 1.0x. `aus` on the exact svd
    polar logs the same numbers.
- **Step time**, plain step, 4 GPUs, `gram_polar_express` 6:

  | rule | mis-8b-proxy | mis-8b |
  |---|---|---|
  | muon | 23 ms | 86 ms |
  | aus | 75 ms | 170 ms |
  | hyperball | 49 ms | 138 ms |
  | elr | 44 ms | 137 ms |
  | sso | 299 ms | 1120 ms |

Every normalisation divides by `norm + presets.EPS` (1e-30, a constant). Where a
row's or unit's sum of squares is below fp32's smallest normal (1.2e-38, a
norm below 1.1e-19), the result is 0 instead (`kernels/floors.py`). Without
it, a moment decayed to entries below ~2.6e-23 (~420 steps without a gradient
at momentum 0.1) has a norm that reads 0: row rules returned updates ~1e6 too
large and the orthogonalisers NaN.

**Adding an optimizer** is one `Compute` subclass and one preset, with no change
to `optimizer.py` or `transport.py`. One that needs the full momentum and
weight of a sharded unit, keeps state and returns the new weight declares
`kind = "unit"`, `needs_weight = True`, `has_state = True`, `output = "weight"`
(and a `state_shape` if its state is not one scalar per unit), and implements
`run(x, w, state, ctx) -> Result` on a `[B, m, n]` stack; `tests/unit_tests/gpu/disco/test_rule_extension.py` does
exactly this and checks it against a reference on every family.

## Families: where a unit is whole

`layout.classify` picks a family from the declared kind, the rule's compute kind
and the placement:

| family | when | communication per step |
|---|---|---|
| `local` | `rows` / `gram` computes, vectors, scalars | none; one packed all-reduce per reducing phase, one for all Grams |
| `expert` | `expert_batch` (`Shard(0)` on the FSDP axis asserted; with more shard ranks than experts some ranks hold none) | none: one batched call per shape block across layers, moments read in place |
| `sharded` | a `unit` compute, rows split over the FSDP axis | one packed all-to-all pair: momentum rows to the owners, updates back |
| `replicated` | whole on every rank (pure DDP) | owners compute, one all-gather of the updates |

Owners are dealt per unit, largest orthogonalisation cost `m n min(m, n)`
first, in a snake over the ranks (`transport.deal`), so a fused parameter's
units land on different ranks; the step's matrix units are dealt apart from
the units that travel on logging steps only. Per-peer row tables come from DTensor's own
chunking, which also covers the nested `_StridedShard` FSDP-over-EP puts on
experts.

## The step

Time runs down; the right lane is the sharded units' traffic, which overlaps
the rest. The MoE bias hook runs before, as a step pre-hook of the container.

```
  momentum    buf <- (1 - m) buf + m g            one compiled pass over all moments
  pre         phases on the rows each rank holds  reducing ones: one packed all-reduce
  send  ----------------------------------------> all-to-all: rows -> owners
                                                   (+ fp32 W where a rule needs it,
                                                    or on a logging step)
  local       rows / gram rules on the local rows      |
              (gram: one packed [n, n] all-reduce)     |  in flight
  experts     one batched call per shape block         |
  sharded     wait <-----------------------------------+
              owners: compute, post, scale per [B, m, n] stack
              send back -------------------------------> all-to-all: owners -> rows
  replicated  owners compute, one all-gather           |
  post        reducing post phases of local units      |  in flight
  [logging]   metrics on the owners, one gather        |
  apply       local, expert and replicated units       |
              wait <-----------------------------------+
  apply       sharded units
```

Rows and unit computes that return a plain update are compiled once per shape
batch, with their row-wise post phases and scale; Gram computes and coefficient
or weight outputs run eagerly. The compile limits are raised once per step,
not per call.

## Memory: what persists and what is temporary

Between steps DiSCO keeps the momentum, the radial accumulators
(`radial_state`), per stateful rule its `rule_state`, and, per sharded
exchange, two buffers of this rank's share of the travelling units. One serves
as the send buffer and as the owner's whole units, the other as the receive
buffer and (fp32 transport) as the holders' updates: their lifetimes do not
overlap, and the copy plans over them are built once. With bf16 transport the
holders' updates are a separate fp32 buffer. An owner holding exactly one unit
computes straight in the receive buffer. The replicated family (pure DDP)
keeps a send buffer, a world-sized receive buffer and its stacks, and applies
an fp32 single-unit parameter's update straight from the gathered buffer.
Everything else a step allocates is freed before `step()` returns.

On a logging step the embedding and the LM head are gathered whole onto their
owner, one at a time, into their arrival buffers (weights, raw moment and
update: three copies), and released before the next. The float64 used by the
spectrum metrics is chunked under 1 GiB. Measured on mis-8b-proxy (dp_shard 4):
persistent exchange memory 0.068 GiB per GPU; logging-step peak 4.39 GiB on the
vocabulary owners, 3.36 GiB elsewhere.

## Checkpoints

Saved per parameter: `momentum_buffer`, `radial_state` (the radial metrics'
accumulators, one row per unit for a fused parameter) and, for a stateful rule,
`rule_state` (one row per unit). Units live on their owner, so `state_dict()`
all-reduces those rows from the owners first. `t` (the step count) is saved
with the param groups; `rule`, `args`, `weight_decay` and `decoupled_wd` are
not: a resumed run follows its own config (a differing value in an older
checkpoint is logged and replaced), and the container supplies them to
upstream's flat loader. Each expert
block's moments are views of one buffer; `state_dict()` hands out copies (DCP's
save staging cannot deep-copy a DTensor over an offset view) and a load copies
them back. A resume is bitwise with the uninterrupted run, AUS included.

## Metrics

**Arming.** DiSCO computes the metrics inside `step()`, where every unit is
whole on some rank, so a step has to be armed before it runs. The trainer calls

```python
rows = disco.begin_step(step, should_log=..., is_final_step=...,
                        head_input=engine.take_head_input_rows)   # before the forward
engine.keep_head_input_rows(rows, seed=step)
... forward / backward ...
grad_norm = engine.optimizer_step()
metrics = disco.collect_metrics(step)
```

`begin_step` arms the MoE router metrics on logging steps and the norms on the
`log_norm_freq` cadence (step 1, steps that are multiples of N, the last
step; N must be a multiple of `metrics.log_freq`, since norms only come on
logging steps). On a norm step with the head in DiSCO it returns how many
rows of the head's input to keep; the container's `step()` takes them from
`head_input` and DiSCO drops them after that step. `collect_metrics` returns
the norms only if the step was armed for them, plus the router metrics.

**Who logs.** Each unit's metrics are computed on its owner. By default one
flat fp32 buffer per rank is gathered to the logging rank. With
`metrics.save_all_shard_ranks` and `metrics.save_for_all_ranks` and a logger
enabled (`trainer.logs_metrics_locally`, decided from config so every rank
agrees on the collective), every shard rank logs its own slice and the gather
is skipped. `distributed.utils.rank_owns_metrics_shard` is the single rule for
which ranks own a slice. Each logging rank then writes its own W&B run,
`<name>-rank<N>` (id `<id>-rank<N>`), all in one group -- `WANDB_RUN_GROUP`,
else the run name, else the run id, else the dump folder's name -- which the
W&B UI folds into
one row; each per-parameter key comes from exactly one rank, so the group's
line is that rank's values. The runs are written offline on the compute nodes
and uploaded as usual: `wandb sync <dump>/tb/*/rank_*/wandb/offline-run-*`.
Launchers must export `WANDB_RUN_GROUP`; `WANDB_GROUP` is read by neither
wandb nor torchtitan. The sink stores a rank's per-unit metrics when
`stores_norms`; `stores_replicated_norms` (shard rank 0) gates only
`scalar_param_supremum/{name}`, which every rank could report. Getting these
wrong drops or duplicates keys, which
`tests/unit_tests/gpu/disco/test_metrics_union.py` checks in both modes.

**Keys.** `track_update_{norm}/{name}` (the applied step), `track_param_{norm}/{name}`
(the post-update weight), `track_radial_{metric}/{name}`, `track_gram_{metric}/{name}`,
`head_{metric}/{name}`, `scalar_param_supremum/{name}`; experts as
`ep_{e}/{name}` with the expert's global index; a Gram metric of a tall unit
(taken over its columns) as `{name}.T`; `track_spectrum_{update,param}/{name}`
only reach the Parquet export. The embedding's norms are taken on its
transpose `[dim, vocab]`, as for a `d_in x d_out` matrix.

**Norms** (`norms_to_log`): `rms_to_rms`, `l1_to_rms`, `rms_to_inf`, `supremum`,
`condition_number`, `frobenius_norm`, `average_entry_size`, `stable_rank`,
`effective_rank`, `effective_rank_squared`; `"default"` is the first five. The
requested set picks the cheapest tier: no SVD (`l1_to_rms`, `rms_to_inf`,
`supremum`, `frobenius_norm`, `average_entry_size`); sigma_max only
(`rms_to_rms`, `stable_rank`; cuSOLVER `gesvda`, exact for sigma_max, falls
back to `gesvd` if it fails); the full spectrum (`condition_number`,
`effective_rank*`), from `sqrt(eigvalsh(W W^T))` of a float64 Gram over the
smaller dimension. It is faster than fp32 `svdvals` -- the cost is `eigvalsh`
on the small Gram, and forming the Gram in fp64 is itself faster than in
non-TF32 fp32 on Hopper, which has fp64 tensor cores -- and far more accurate
on sigma_min
(2.7e-11 vs 2.8e-7 relative at condition number 1e3); no rescaling makes fp32
good enough, since `kappa(W W^T) = kappa(W)^2`. `condition_number` is in
`"default"`, so default costs what `"all"` costs: on mis-8b at dp_shard 64 the
norm pass is ~170 of ~200 ms of metric compute per rank per logging step,
almost all of it the spectrum norms. Environment overrides: `DISCO_SVDVALS_DRIVER`,
`DISCO_SVDVALS_DRIVER_SIGMA_ONLY`, `DISCO_SPECTRUM_BACKEND`.

**Radial** (`helpers/radial_helper.py`, always on with the norms): 26 scalars on
how a weight's whole-tensor norm and direction move each step, from the
pre-update weight and the post-update weight: `radius`, `raw_step`,
`relative_step`, `radial_cosine`, `tangent_fraction`, `angle` (via `atan2`;
`angle_from_cos` is the ill-conditioned `arccos` form, kept as a check),
first/second-order radial terms, `E_radial`, the running accumulators
`raw_A2`, `angular_A1`, `angular_A2`, `R1` (saved in `radial_state`), the
spectral counterparts (`spectral_radius`, `spectral_growth`, ...), the
radialities under three operator norms, and `aus_*`, the angle of the step
between the normalised pre- and post-update weights in four norms (Frobenius as
`2 sin(angle/2)`; the axis norms form the difference explicitly, because the
reduction-expansion forms lose 12-50% in fp32 at a 5e-4 step). The spectral
ones use the exact top singular pair of the pre-update weight from the float64
Gram's `eigh` (`norm_helper.gram_top_singular_pair`). A step smaller than
`1e-6` of the weight reports the deterministic degenerate values, so DDP and
FSDP agree. Every metric is present for every input (fixed arity, 0 when
undefined): the flat buffers are sized from the name list.

**Gram** (`helpers/gram_helper.py`, `gram_level` 1-3, cumulative): geometry of
the pre-update weight, the raw moment and the post-update weight, oriented over
the smaller dimension (so the vocabulary matrices give `[dim, dim]`, never
`[vocab, vocab]`). Level 1 is entrywise (~4 ms per `[768, 2048]` matrix), level
2 adds eigenspectra (~74 ms), level 3 whitened and cross geometry (~125 ms).
Vector-valued entries go only to the Parquet export.

**LM-head geometry** (`helpers/head_geometry.py`, `head_{metric}/{name}`, on the
norm cadence): `eta`, the softmax-quotient step ratio `rho_{mean,rms,max}`, the
row-geometry factor `gamma_H` and its probe-free `gamma_H_sweep`, `a_H_*`,
`U_rms_to_inf`, `h_rms_mean` and LAR (arXiv 2605.28975) for update and weight.
Computed on the head's owner from the whole pre-update weight and the applied
direction; the one input the optimizer cannot produce is the head's input
`h`. Under the chunked loss (opt_moe's) the model's forward stops before the
head and returns `h`, so the engine keeps `head_diag_probes` sampled rows of
the last microbatch's output (`training_engine.keep_head_input_rows`): no
second forward, no hook, nothing inside the compiled model, no collective.
Off when no DiSCO instance holds the head, without the chunked loss, under PP,
and under CUDA graphs (the capture runs in the eager forward body).

**MoE router metrics** come from `moe_balance.py`'s hook, which updates the
router bias every step and publishes the metrics only on armed steps.

**Cadence.** `dense_norm_steps: N` also logs the norms every step up to N
(needs `metrics.dense_log_steps` at least as large). `effective_rank_freq: N`
computes `effective_rank`, the one norm that needs the whole spectrum, only
on norm steps divisible by N (and steps 1 and last). `spectral_norm_freq: N`,
a multiple of `log_norm_freq`, does the same for every SVD-based metric
(`rms_to_rms`, `stable_rank`, `condition_number`, `effective_rank*`, the
spectra, the spectral radial metrics); the SVD-free norms keep
`log_norm_freq`.

**Model statistics** (`optimizers/model_stats.py`, off by default, on the
`log_norm_freq` cadence):
- `log_residual_stats`: the RMS of every block's input and residual write and
  of the final norm's input, taken in opt_moe's eager layer loop between the
  compiled blocks (the blocks' graphs are the same either way).
- `log_gain_stats`: each RMSNorm gain's mean, RMS and `||dg|| / ||g||`; with
  column gains (`amphione` `columns`, `angular_muown` `col_scale`) also each
  unit's c (`col_gain_*`: rms, mean log|c|, q10 / q50 / q90, relative update),
  read from the rule state.
- `log_sink_stats`: every attention sink value,
  `sink_value/<layer>.attention/head-<h>` (needs a model with sinks).
- `log_gram_input`: on spectral norm steps, sigma_1 / ||X||_F of every tensor
  `gram_polar_express` receives, per unit (`gram_input_s1f/<unit>`), and the
  count of units in [0.94, 0.955], where the iteration fails at
  `polar_express_safety` 1.01 (`disco/gram_input_stats.py`).

## Orthogonaliser backends

| backend | notes |
|---|---|
| `gram_polar_express` | the production choice (6 steps). Iterates on the small symmetric Gram with quack's symmetric-GEMM kernels, work dtype fp16 (the iteration squares its operand, so mantissa beats range: error 0.0027 fp16 / 0.0132 bf16 / 0.0017 fp32). quack's kernels segfault on other dtypes, so for any other work dtype the kernel falls back to plain torch matmuls (correct, slower) |
| `polar_express`, `polar_express_triton` | Polar Express coefficients; honour `steps`. Error vs the exact polar factor: 0.36 at 4 steps, 0.09 at 5, 0.006 at 6, flat after |
| `newtonschulz5`, `newtonschulz5_triton` | the classic quintic iteration; both run the same math |
| `svd` | exact polar factor, the reference; directions below LAPACK's rank tolerance are zeroed, not sent to 1 |
| `identity` | no orthogonalisation (as a compute, `identity` is the row map: plain momentum SGD) |

The triton backends are autotuned, so they are not bitwise reproducible across
runs; bitwise tests use `svd` or `gram_polar_express`.

**Polar express safety factor** (`args.polar_express_safety`, default 1.01).
The polar express backends (`gram_polar_express`, `polar_express`,
`polar_express_triton`) evaluate every triple but the limit at x / f:
`(a / f, b / f^3, c / f^5)`; the input normalisation stays at 1.01. 1.01 is
the paper code's value and today's kernels to the bit; upstream's Gram
Newton-Schulz uses 1.05, and the optimizer study ran `gram_polar_express` at
1.05 (its configs spelled the key `safety`, which now fails as renamed).
`newtonschulz5` and `svd` have no such factor and refuse it. A row group that
inherits it from the default ignores it.

## Weight averages (`averages`, off by default)

`components/optimizer/averages.py`, a sibling of EMA in the training engine.
After step s (counted from `start_step`, every `every` steps) the power
average with exponent gamma is `avg <- avg + (s^gamma / S) (w - avg)`, S the
sum of s^gamma so far (gamma 0: the plain mean).

| key | meaning |
|---|---|
| `gammas` | the exponents, e.g. `[0, 1, 3, 7]`; empty: none |
| `start_step`, `every` | when the updates fire |
| `renorm` | `[regex, mode]` pairs on canonical FQNs, first match wins: `frobenius` / `spectral` scale each unit of the average to the live unit's norm (the radius a rule holds), `rows` to the averaged row lengths, `none` |
| `evaluate` | score every average with the validator whenever the trainer validates (`avg_g<gamma>`) |
| `snapshot_steps`, `snapshot_folder` | write the full weights (fp32, canonical FQNs) after these steps |
| `export_steps` | write each renormalised average after these steps |
| `score_files` | score these weight files (e.g. a mixture of snapshots) with the validator at start, then stop |
| `start_from_gamma` | right after the run's first load of `initial_load_path` (not on a restart from its own checkpoint), replace the weights by this gamma's renormalised average and stop averaging; the optimizer state is the checkpoint's |

## Validation sequence length

`validator.fixed_seq_len` (default None: the training `max_context_length`,
so existing configs are unchanged) scores every run on sequences of that
length, so runs with different training context lengths are compared on the
same sequences. The RoPE caches cover it when it is longer than the training
context.

## NaN probe (debugging, inert by default)

`torchtitan/observability/nanprobe.py` is gated by environment variables and does nothing
unless `NAN_PROBE_FROM` (first step) and `NAN_PROBE_DIR` (output folder) are
set. From that step it writes, per rank, summaries of each tensor (count of
non-finite entries, largest finite magnitude, RMS, worst row) at the
microbatch losses, after the backward, after the update, and inside DiSCO
(effective gradients, every rule call, the updates), plus no_grad forward
probes of the step's microbatch; a backward probe runs only on a step that
is already non-finite (the trainer's assert follows). Its hooks sit in
`training_engine.py`, `trainer.py` and `disco/optimizer.py`. `NAN_PROBE_STOP` (last step),
`NAN_PROBE_BIG` (the magnitude it flags, default 1e4), `NAN_PROBE_EAGER`
(default 1) and `NAN_PROBE_FORCE_BAD` (a smoke test of the bad-step path)
tune it.

## Tests and benchmarks

```bash
source llm_env/activate.sh; cd resources/torchtitan
# rules, groups, transport, cadence, head input (CPU, minutes)
CUDA_VISIBLE_DEVICES= python -m pytest tests/unit_tests/cpu/optimizers -q
# every family and rule against a reference written from the equations, 4 gloo ranks
CUDA_VISIBLE_DEVICES= python -m pytest tests/unit_tests/gpu/disco -q
# metric helpers against brute-force references
bash tests/unit_tests/disco_metrics/run_all.sh
# optimizer step time on 4 GPUs (plain and logging steps; --profile N adds a GPU-time breakdown)
torchrun --nproc_per_node 4 tests/unit_tests/gpu/disco/bench_step.py \
    --flavor mis-8b-proxy --steps 30 --log-norm-freq 10 --dp-shard 4
```

`run_all.sh` includes `verify_norm_helper`, which needs a GPU (cuSOLVER's
`driver=`), so it fails under `CUDA_VISIBLE_DEVICES=`. Keep
`CUDA_VISIBLE_DEVICES=` for the pytest suites on the login node (it has a GPU),
run the 4-rank
suites one at a time (concurrent suites collide on ports), and build FSDP in a
CPU test on an explicit CPU mesh.

## Not supported, and known limits

- **TP inside DiSCO** (`assert tp == 1`). If needed it gets its own transport:
  one exchange group over the flattened `dp_shard x tp`, colwise TP first.
- **Conv1d** (rank 3 must be declared; no `conv` kind yet). A depthwise conv
  `[C, 1, K]` views as a `[C, K]` unit, so its kernel size is the unit's `n`.
- **Pure DDP with MoE**: supported, but its logging-step expert temporaries
  were not optimised (production runs FSDP).
- **Logging steps** pay a blocking device-to-host copy in the MoE publish,
  which exposes DiSCO's kernel launches after it (~70 ms of GPU time measured
  on mis-8b at FSDP 16).
- **Normalised rules and idle rows**: a row whose gradient stops keeps taking
  full-size steps along its decaying momentum for ~300 steps (rare embedding
  rows, idle experts). Measured, loss effect unmeasured.
- **PP**: supported by the optimizer (a group that matches nothing on a stage
  is dropped there), but without the LM-head geometry.
- **Gram rules on sharded units** (`lm_head`, `right_polar` on a sharded head
  or embedding) issue a blocking all-reduce behind the async all-to-all, so
  the expert compute waits for the exchange. Not the production rules.
- **W&B resume / fork** (`WANDB_RESUME_FROM`, `WANDB_FORK_FROM`) get no rank
  suffix under per-rank logging.
- The metrics gather sends to every rank; only the logging rank reads it.
- **Rule state is not sharded under FSDP (to confirm before production).** For every
  unit outside the experts, a stateful rule's `rule_state` is allocated whole on
  every rank (`DiSCO._init_state`): one row per unit, of `state_shape(m, n)`. Only
  the unit's owner writes its rows; the others stay zero, and `state_dict()` sums
  the rows over the ranks to save them. Numbers and checkpoints are right (resumes
  are bitwise), but the memory is not split across ranks:
  - `angular_muown` / `onerate` / `amphione` keep `(m, n + 3)` per unit (the
    direction's momentum and the row lengths' state), `lie_relative` `(m, n)`: a
    full fp32 copy of every hidden matrix on every rank. 336M (335M hidden
    parameters): about 1.35 GB per rank, where 1/8 would be about 0.17 GB; an 8B
    dense model about 28 GB.
  - `aus` and `sso` keep a length-n vector per unit: replicated too, but small.
  - These groups (`momentum: 1.0`) still get the generic `momentum_buffer`
    (sharded like the parameter), which they never use.
  - The experts' `rule_state` is a DTensor, sharded like the parameter.
  The fix: allocate only the rows of the units a rank owns, and no
  `momentum_buffer` when the rule keeps its own momentum; both change the
  checkpoint path and need the FSDP resume tests (`gpu/disco/test_step_paths.py`).

## Gradient-noise logging (`optimizer.gnoise`, off by default)

`optimizers/gnoise.py` (E1), `gnoise_research.py` (research mode), `gnoise_math.py` (the
estimators). Off by default; when off nothing is installed and nothing runs.

**When.** E1 runs on DiSCO's norm-logging steps: by default every one (step 1, every step
<= `dense_norm_steps`, every `log_norm_freq`-th step, the last step). `e1_freq` N (a multiple
of `log_norm_freq`) keeps the steps divisible by N; `e1_dense_phase` false drops step 1, the
dense phase and the last step. `research_freq` and the curvature frequencies must be multiples
of the E1 interval (`e1_freq`, or `log_norm_freq` when 0). The finalisation is vectorised over
all units, rows, heads and types (static index maps); metrics are kept as vector families and
gathered by owner with one sync.

**Small batch.** One (rank, microbatch): torchtitan reduces the gradient after every
microbatch, so that is the smallest gradient that exists. For a routed expert it is one
(expert copy, microbatch): EP 1, the rank; EP > 1, its EP group. K = dp ranks x accumulation;
K_e = (efsdp x dp_replicate) x accumulation. K < 2 is refused at set-up.

**Where it reads.** A per-instance wrapper on every `FSDPParamGroup.post_backward`, at its
entry: the group still holds the unsharded weight the forward used and this small batch's
gradient, before FSDP reshards and reduces (torch `_fsdp_param_group.py`, pinned by a test).
The reduced gradient G is read before the clip. One small batch per (param, microbatch),
counted exactly once; a group still sharded at the entry, or a skipped reduce outside the
chunked loss's chunks, raises.

**Estimators.** With n_b valid tokens, g_b the gradient of sum(loss) / N_den, ghat_b =
(N_den / n_b) g_b, N = sum n_b, Ghat = (N_den / N) sum_b g_b, for a projection P fixed with
respect to the batch: A = mean_b ||P ghat_b||^2, F = ||P Ghat||^2, H = mean_b 1 / n_b,
tr = (A - F) / (H - 1/N), S = (F H - A / N) / (H - 1/N). Small batches with no token are left
out; a component with fewer than two is not estimable (NaN). P is built from the compute
copy of W (the fp32 master cast to FSDP's param_dtype on the F side, bitwise the same values).
A basis from G_t or M_t is not offered: the unbiasedness needs P independent of the batch.

**Caveats.**
- Sampling: the loader packs documents (concat_then_split, no shuffle), so the estimate is
  the noise per rank-microbatch. It scales to tokens only as far as microbatches are
  independent draws; the accumulation test (mb 32 vs 16 x 2) measures the gap.
- Experts: unbiased conditional on the routing, treating the tokens routed to an expert as
  draws from its routed-token distribution; n_b,e is the routed count (every routed token, a
  masked label included), N_e = sum_b n_b,e, Ghat_e = (N_den / N_e) G_e. B_simple is in routed
  tokens and, x N / N_e, in total tokens. Small batches with fewer routed tokens than min(m, n) are rank-deficient
  (counted per step: `gnoise_ex_lowtok`).
- Per-step SE: the delete-one jackknife needs <P ghat_b, P Ghat> per b. E1 has it for rank-1
  components (Frobenius / spectral radial); research mode for every component.
- Router: covered with no aux loss or the sequence-wise one; left out under the batch-wise one
  (its gradient is no per-token sum). The expert bias is a buffer, not covered.

**Research mode** (`research_freq` > 0). Every rank copies its small batch's gradient of the
covered units (bf16) into a send buffer laid out by owner; one all-to-all over the dp world
moves them (a dense matrix whole to one owner, a vocabulary matrix by row blocks, an expert
unit only among the ranks holding that expert). The owner then has all K small batches and W:
exact jackknife SEs, the spectral radial (u1 v1^T of W_t), top-k shares (W_t's top-k on the
smaller side; A1 predicts k / min(m, n)), and the LMO agreement a_s = mean over disjoint pairs
of <O_S, O_S'> / min(m, n) at every s dividing K with s <= K / 2 (one partition per size from a
private generator seeded by the step, three at the half, all three pairings at K = 4).
- Exact polar = the eigendecomposition of the fp64 Gram; singular values below 1e-6 sigma_max
  map to 0 (resolvable to about 1e-8 sigma_max). It is the exact polar, not the training
  backend's Polar-Express LMO, and it is the same for every training rule: research reads no
  optimizer state (vectors: sign, embedding / head / router rows: row-l2, matrices: polar).
- E a_s = kappa(s n_b)^2 under the linear-drift model (two independent halves); reading the
  momentum step's kappa from it assumes the noise-dominated law kappa ~ sqrt(P B).
- Memory on a research step: about 4 B x (P_dense + P_experts / EP) x accumulation per rank
  (snapshot + received), freed after the step.

**Curvature** (`curvature_model_freq`, `curvature_matrix_freq`; research runs): same-batch,
after the optimizer step: c = 2 (L(W_{t+1}) - L(W_t) - <G_t, Delta>) / ||Delta||^2 from one
forward, and per unit <Delta_k, G'_k - G_k> / ||Delta_k||^2 from a second forward and backward.
All model buffers (MoE counts, aux-loss accumulators) and the RNG are restored and the
gradients cleared.

**Update metrics** (hidden matrices: layout kind `matrix`, no experts). DiSCO calls
`update_observer` at the end of `step()` with the update u it applied (W <- decay W -
lr u, alpha W - u, or W - u for rules that return weights); O_t = u. On every E1 step
`gnoise_psi/<unit>` = <O_t, G_t> / (|O_t| |G_t|), G_t the reduced gradient (read after
the clip, a common scalar). On the E1 steps divisible by L, the smallest multiple of the
E1 interval >= 8, `gnoise_lag<h>/<unit>` = <O_t, O_{t-h}> / (|O_t| |O_{t-h}|) for h = 1,
2, 4, 8; the local rows of O are copied only on the steps L - h before (at most four
copies per parameter, dropped once read; NaN after a resume until refilled). Research
steps add `gnoise_psi_half` / `gnoise_psi_quarter` (G from the subset sums of the groups
{b : b mod g = a}, g = 2 / 4, b the rank-major small-batch index; mean over a; the
small-batch gradients are kept on their owner through the optimizer step and O is
gathered whole), and per matrix unit `gnoise_ntop<k>` (k = 1, 2, 3, k <= K - 1: the
top-k eigenvalue share of the K x K Gram of sqrt(n_b) d_b, d_b = ghat_b - Ghat, i.e. of
the per-token noise covariance; an isotropic noise gives about k / (K - 1)) and, when K
>= 8 (else one startup line), `gnoise_bs1` = `gnoise_bs1_noise` / `gnoise_bs1_grad` =
||C_row^(1/2)||_S1^2 / ||Ghat||_S1^2, C_row = sum_b n_b d_b d_b^T / (K - 1) over the
unit's rows (Naganuma et al. 2602.03001 Eq. 17; ||Ghat||_S1 includes the noise's
upward bias).

**Refused (raise at config time, only when enabled):** TP, CP or PP > 1; CUDA graphs;
CPU offload; compiled autograd.
