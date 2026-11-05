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

Momentum convention: `momentum` is the memory weight beta, as Muon's, in the
averaged form `buf <- beta * buf + (1 - beta) * g` (the production recipe:
0.95, the head 0.99). `nesterov: true` hands the rule
`beta * buf + (1 - beta) * g` instead of `buf`. `momentum: 0` is the raw
gradient (no buffer is updated); a momentum outside [0, 1) is refused.

## Package layout

```
optimizers/
  container.py      DiSCOOptimizersContainer: config, param groups, arming, metrics hand-off
  moe_balance.py    MoE load-balance bias hook; router metrics on logging steps
  model_stats.py    residual, RMSNorm-gain and sink statistics (log_*_stats)
  utils.py          FQN cleanup, the DiSCO kwargs the container passes
  disco/
    optimizer.py    DiSCO: build (families, exchanges, metric registry), step(), state
    layout.py       families (classify), per-rank row tables
    rules.py        what a rule is: Phase, Compute, Chain, Rule, Result; the registry
    presets.py      the built-ins: pre phases, computes, presets sgd/muon/hyperball/
                    amphione
    transport.py    Exchange (one packed all-to-all pair), Replicated (all-gather)
    metrics.py      MetricSink: per-unit registry, one flat buffer, one gather, unpack
    gram_input_stats.py  sigma_1 / ||X||_F of gram_polar_express's inputs (log_gram_input)
  kernels/          orthogonaliser backends: zeropower, triton Newton-Schulz,
                    Gram Newton-Schulz, underflow floors
  helpers/          norm_helper, gram_helper, radial_helper, head_geometry, power_iteration
  export/           spectrum / gram-vector Parquet export (W&B artifacts)
models/common/param_layout.py   the declaration vocabulary models use (ParamLayout, UnitSpec)
components/optimizer/averages.py          weight power averages, snapshots, scoring (averages)
```

## Configuring it

A run states named parameter groups; the production recipe
(the recipe `opt_moe_mis_8b` in `torchtitan_recipes/models/opt_moe.py`, lr = the hidden matrices' lr):

```python
config.optimizer.groups = {
    "gains": {"match": r"(^|\.)(q_norm|k_norm|attention_norm|ffn_norm|norm)\.weight$",
              "rule": "sgd", "args": {"pre": "identity", "compute": "sign", "scale": "one"},
              "lr": 0.5 * lr, "momentum": 0.95, "nesterov": True, "weight_decay": 0.0},
    "default": {"rule": "amphione", "lr": lr, "momentum": 0.95, "nesterov": True,
                "weight_decay": 0.0,
                "args": {"scale_nesterov": False,
                         "compute": "gram_polar_express", "steps": 6,
                         "polar_express_safety": 1.05,
                         "frozen_scale_units": ["feed_forward.router.gate"]}},
    "embedding": {"match": r"tok_embeddings\.weight", "rule": "sgd",
                  "args": {"compute": "row-l2", "scale": "sqrt_n", "pre": "identity"},
                  "lr": 0.001953, "momentum": 0.95, "nesterov": True, "weight_decay": 0.0},
    "lm_head": {"match": r"(output|lm_head)\.weight", "rule": "sgd",
                "args": {"compute": "row-l2", "scale": "inv_sqrt_n", "pre": "identity"},
                "lr": 0.125, "momentum": 0.99, "nesterov": True, "weight_decay": 0.0},
}
```

- `default` is required; every other group inherits from it and selects its
  parameters with exactly one of `match` (an FQN regex) or `kind` (a declared
  kind, e.g. `{"kind": "unembedding"}`). Groups are tried in order, first match
  wins, the rest go to `default`.
- Group keys: `optimizer` (default `"DiSCO"`; `"AdamW"` sends the group to an
  AdamW beside DiSCO), `lr`, `weight_decay`, `momentum`, `nesterov`, `rule`,
  `args`. Every key the config file sets is a CLI flag:
  `--optimizer.groups.default.lr 0.05`, `--optimizer.groups.default.args.steps 8`.
- Weight decay scales W by `1 - lr weight_decay` per step. The rule's own
  step is unaffected. `lr`, `weight_decay`, `momentum` and `nesterov` come
  from the config, never the checkpoint, so a resume may change them (one
  limit: see "Checkpoints").
- `args` merge over the default group's. A group with the default's `rule`
  takes all of the default's args; a group with another rule takes only the
  ones its preset accepts, minus the preset's `own` ones (below). So a group
  whose rule differs must state every arg that should not come from the
  default: under a `muon` default with `compute: gram_polar_express`, a bare
  `{"rule": "sgd"}` group inherits that orthogonaliser.
- Build errors: unknown keys and args (old spellings such as `backend` among
  them); DiSCO-only keys (`rule`, `args`,
  `momentum`, `nesterov`) on another optimizer's group; a
  DiSCO with no default `lr`; `weight_decay` in a `hyperball` or an
  `amphione` group (a rule that keeps its own momentum);
  `gram_level` outside 0-3; an iterative compute with `steps < 1`; the
  vocabulary matrices under a gathering rule; a Gram compute on a wide unit;
  declared units that overlap or misstate their shape; a stateful rule whose
  state shape differs across one expert parameter's units (the others keep one
  state per unit, any shape).
- `opt_moe_mis_8b()`'s groups are a valid run on their own: the production
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
| compute | `compute` (+ `steps`) | the only place a direction is formed. Row maps `identity`, `row-l2`, `sign` (no communication: the rows a rank holds are enough); orthogonalisers `newtonschulz5`, `newtonschulz5_triton`, `polar_express`, `polar_express_triton`, `gram_polar_express`, `svd`; `gram_polar` (the polar factor through the `[n, n]` Gram: one all-reduce, no all-to-all); chains `main+row_map+...` (`gram_polar+row-l2`) |
| post | (preset-internal) | phases that need a statistic over all rows after the compute |
| scale | `scale` | the unit's step relative to `lr`, from its shape `[m, n]` = `[d_out, d_in]`: `one`, `spectral` (sqrt(m/n)), `image_spectral`, `n`, `sqrt_n`, `inv_n`, `inv_sqrt_n`, `sqrt_mn`, `rms_parity` (min(sqrt(m/n), 1): unit rows at the Frobenius norm of polar x spectral). A constant per unit, the last multiply of the compiled compute (for a Gram compute, or after a reducing post phase, it runs after it) |

A `Compute` declares where the unit must be whole (`kind`: `rows` = the rows a
rank holds, `gram` = those rows plus one `[n, n]` all-reduce, `unit` = whole on
its owner), whether it needs the pre-update weight (`needs_weight`), whether it
keeps state (`has_state`, shape `state_shape(m, n)`) and what it returns
(`output`). One apply serves every output: `"update"` ->
`w <- decay w - lr U`; `"update_coeffs"` -> per unit `w <- alpha w - U`, the
step's size folded into U (alpha rides one small all-reduce); `"weight"` ->
`w <- w - (W - W_next)` (the step `W - W_next` travels in the update's slot;
equal to W_next up to rounding). The apply
adds lr and weight decay only to a plain update: a coefficient or weight
output must fold them in itself, from the `ctx` it is given
(`StepCtx(lr, wd, t, momentum, nesterov, frozen, colmask)`, whose `decay` is
`1 - lr wd`; `momentum` / `nesterov` are the group's, for a rule that keeps
its own averages; `frozen` / `colmask` the stack's unit masks),
and cannot have post phases or a scale. A contract that cannot hold fails
when the rule is built, not at the first step.

Presets, each a function registered with `@register(name, own=...)` whose
keyword arguments are its args and defaults:

| `rule` | pipeline | defaults | `own` |
|---|---|---|---|
| `sgd` | `pre` -> `compute` -> scale | identity, identity, steps 0, `one` | |
| `muon` | `pre` -> orthogonaliser -> scale; per block with `block_rows` / `block_cols` | identity, `newtonschulz5`, 5, `spectral`, blocks 0 | |
| `hyperball` | `FrobeniusAUS`: `W+ = r N(W - s r N(U))`, r = `\|\|W\|\|_F` | `newtonschulz5`, 5; `tangent` false, `exact_angle` false | |
| `amphione` | `AmphioneMap`: `W = Diag(g) U`, unit rows turned by the orthogonalised momentum, row scales by a sign step, one rate: the group's lr from the trainer's scheduler, one momentum: the group's; optional column scales (see "Amphione" below) | `gram_polar_express` 6, `scale_nesterov` false, columns off | `frozen_scale_units` |

The production recipe: `amphione` on the hidden matrices, `sgd` with `compute:
sign` on the RMSNorm gains, and `sgd` with `compute: row-l2` on the embedding
(`scale: sqrt_n`) and the head (`scale: inv_sqrt_n`). A rule with a whole-unit
direction (`hyperball`, `amphione`) takes a row compute as that direction (the
attention's conv); it refuses a Gram compute. Every group states its own
`rule`: a group without one inherits the default group's.

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
"conv": {"kind": "conv", "rule": "amphione", "args": {"compute": "row-l2"}},   # the recipe
"conv": {"kind": "conv", "rule": "sgd", "args": {"compute": "row-l2"}},        # row-norm
"conv": {"kind": "conv", "rule": "muon", "args": {"compute": "polar_express", "steps": 5}},
"sinks": {"match": r"\.sinks$", "rule": "sgd", "args": {"compute": "sign", "scale": "one"}},
```

- **Amphione with `row-l2`** (the recipe): every channel's filter is its own
  `1 x K` matrix, whose spectral normalisation is its l2 norm; unit rows, the
  rescale, the row lengths' sign step and the rate are Amphione's.
- **Row-norm** normalises each channel's K taps. It runs on the channels a
  rank holds, so it needs no communication.
- **Orthogonal** is muon on `[C, K]`, gathered to an owner. With K = 4 the
  tiled orthogonalisers cannot run: `gram_polar_express` needs dims divisible
  by 8, and the Triton ones need at least 16. They are refused at build, so use
  `polar_express`, `newtonschulz5` or `svd`. `gram_polar` (the polar factor
  through the `[K, K]` Gram, not yet trained with) orthogonalises `[C, 4]`
  exactly at 8 steps, with a 4x4 all-reduce and no gather.
- **Sinks:** under the production default, the `[1, n_heads]` unit is refused
  (one row). Give them a group: `sign` moves each head's sink by lr;
  `row-l2` normalises the whole vector.
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

**Hyperball** (MuonH, arXiv 2606.16899) needs the weight on the owner and
returns per-unit coefficients. Its step size s is the group's `lr`, the ELR:
the step relative to `||W||_F`.
- `exact_angle: true` solves s in closed form so that the angle between W and
  W+ is exactly lr (arXiv 2607.22444).
- `tangent: true` (MuonH-T) removes the momentum's component along W before
  the orthogonaliser.
- It holds each unit's norm, which undoes weight decay exactly, so its group
  must set `weight_decay: 0` (checked at build).
- A radius-relative step never moves a zero-norm unit; the first step warns if
  one exists.
- Its minus-square-root schedule is the trainer's `lr_scheduler` with
  `decay_type: "sqrt"` and `decay_ratio: 1.0`.

**One schedule.** Every group runs the trainer's `lr_scheduler` on its own
base `lr` (upstream's one `LambdaLR` lambda); a group has no schedule keys of
its own. On a resume the scheduler writes this run's lr into every group, so a
run resumed under a changed schedule runs it from the next step on.

**Amphione** (`rule: amphione`). Each hidden matrix is `W = Diag(g) U`: U has
unit rows, g holds their signed row scales. From the raw gradient G (DiSCO
averages nothing for this rule), with `eta_t = lr` (the group's lr at this
step, from the trainer's scheduler) and `beta` = the group's `momentum`:

```
gamma = rowdot(G, U),  D = Diag(g) G
M <- beta M + (1 - beta) D
O = Orth(beta M + (1 - beta) D)             (the group's nesterov; Orth(M) without)
O <- O ||U||_F / ||O||_F
U <- RowNorm(U - eta_t O)
m_g <- beta m_g + (1 - beta) gamma,  g <- g - eta_t sign(m_g)
```

The group's `nesterov` acts on the direction only; `scale_nesterov: true`
gives the scales' sign steps `sign(beta m_g + (1 - beta) gamma)` too. g starts
at the step-1 row norms (a zero row at sqrt(0.33)); there is no step-1
rescale, so a larger start belongs in the init. Before wave 2 the direction
ran in the sum form `M <- beta M + D, O = Orth(D + beta M)`: the input differs
by the factor 1 - beta, which the polar factor and the rescale drop, so the
step is the same. Args:

| arg | default | meaning |
|---|---|---|
| `compute`, `steps` | `gram_polar_express`, 6 | the direction: an orthogonaliser (`polar_express`, `newtonschulz5`, `svd`, ...), or `row-l2` (`O = rowl2(beta M + (1 - beta) D)`: the conv); a Gram compute is refused |
| `scale_nesterov` | false | Nesterov on the row (and column) scales' sign steps |
| `column_scales` | false | column scales c, `W = Diag(g) U Diag(c)`, on `column_scale_units` |
| `column_scale_units` | wq, wk, wv, w1, w3 | the unit types that learn c (`AMPHIONE_COLUMN_SCALE_UNITS`) |
| `column_scale_lr_mult` | 0.0625 | c's sign step is `column_scale_lr_mult * eta_t` |
| `frozen_scale_units` | () | unit types of the group whose row scales g stay at their start; their rows still turn (the MoE router gate: `["feed_forward.router.gate"]`); each type must name a unit of the group; never inherited from the default group; not with `columns` |

The conv group, at the hidden matrices' lr and momentum:

```python
"conv": {"match": r"(^|\.)attention\.(q|k|v)_conv\.weight$", "rule": "amphione",
         "lr": 0.01562, "momentum": 0.95, "nesterov": True, "weight_decay": 0.0,
         "args": {"compute": "row-l2", "frozen_scale_units": []}},
```

It inherits the default group's other `amphione` args (`scale_nesterov`,
...). With `column_scales: true` on the default, set `column_scales: false` on the conv
group: the conv units are not in `column_scale_units`, and DiSCO refuses a column
unit type that matches no unit of the group. A group inherits the default's
`polar_express_safety` only through a polar express compute of its own (the
conv's `row-l2` and the sgd groups take none); a group that sets it on another
compute is refused.

The rule keeps its own averages, at its group's `momentum` and `nesterov`; the
group sets `weight_decay: 0` (the rule returns the new weight). Its state is
`[M | g | m_g]` per unit (two more rows `[c; m_c]` with column scales). A group config:

```python
"default": {"rule": "amphione", "lr": 0.01562, "momentum": 0.95, "nesterov": True,
            "weight_decay": 0.0,
            "args": {"compute": "gram_polar_express", "steps": 6,
                     "polar_express_safety": 1.05, "scale_nesterov": False}},
```

What the optimizer study ran, where it differs from these defaults: Gram
Polar Express at `polar_express_safety` 1.05; a constant trainer lr with an
eta table 1 - sqrt(t / T), where `amphione` takes the trainer's sqrt schedule;
the row-scale average 0.9 (the direction's 0.95); the row scales x4 at step 1
by the optimizer, measured to match x4 in the init within 0.003 in loss.

**Column scales (`amphione` `column_scales: true`; off by default).** c takes the
sign step above at `column_scale_lr_mult * eta_t` on the `column_scale_units`; the other
units keep c = 1. c lives in rows of the rule state, so it is checkpointed with
it. The parameter is the effective W, so every logged norm and radial metric
is W's; U, g and c are not logged (c's statistics: `log_gain_stats`).

**Step time**, plain step, 4 GPUs, `gram_polar_express` 6:

| rule | mis-8b-proxy | mis-8b |
|---|---|---|
| muon | 23 ms | 86 ms |
| hyperball | 49 ms | 138 ms |
| amphione, the production recipe's groups (GQA, FSDP 4; llm-0.5.4 wave 2, before the E1 / E2 fixes) | - | 136.6 ms |

The last row: synthetic gradients, every group of `opt_moe_mis_8b()`
(amphione on the hidden matrices and experts, sign on the gains, row-l2 on the
vocabulary matrices); peak 58.6 GiB, optimizer state 6.91 GiB per GPU.
mis-30b-a3b on 16 GPUs: 146.3 ms, peak 68.7 GiB.

Every normalisation divides by `norm + presets.EPS` (1e-30, a constant). Where a
row's or unit's sum of squares is below fp32's smallest normal (1.2e-38, a
norm below 1.1e-19), the result is 0 instead (`kernels/floors.py`). Without
it, a moment decayed to entries below ~2.6e-23 (~420 steps without a gradient
at momentum 0.9) has a norm that reads 0: row rules returned updates ~1e6 too
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
  momentum    buf <- b buf + (1 - b) g            one compiled pass over all moments
  pre         phases on the rows each rank holds  reducing ones: one packed all-reduce
  send  ----------------------------------------> all-to-all: rows -> owners
                                                   (+ fp32 W: every unit on a logging
                                                    step; for a rule returning the
                                                    weight on its first step, after a
                                                    load and on the copy checks only
                                                    (the owners keep fp32 copies);
                                                    hyperball every step)
  local       rows / gram rules on the local rows      |
              (gram: one packed [n, n] all-reduce)     |  in flight
  experts     batched calls per shape block, whole     |
              layers up to 1 GiB per fp32 temporary    |
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
batch, with their row-wise post phases and scale; amphione's element-wise maths
around its direction is compiled on CUDA (no column scales; fp32 inside, any
transport dtype); Gram computes and other coefficient or weight outputs run
eagerly. The compile limits are raised once per step, not per call.

## Memory: what persists and what is temporary

Between steps DiSCO keeps the momentum (where it averages), the radial
accumulators (`radial_state`), per stateful rule its `rule_state` on the units'
owners only (amphione: `(m, n + 2)` per unit, about 1/world of the hidden
matrices per rank; an expert block's states and moments as one buffer each),
the owners' fp32 copies of the weights of the units whose rule returns the
weight (about 1/world of those matrices), and, per sharded exchange, two buffers of this rank's share of the travelling units. One serves
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

Saved per parameter:
- `momentum_buffer`, only where DiSCO averages (0 < beta < 1 and the rule keeps
  no momentum of its own; amphione keeps its averages in its rule state). Each
  expert block's buffers are views of one buffer; `state_dict()` hands out copies
  (DCP's save staging cannot deep-copy a DTensor over an offset view) and a load
  copies them back.
- `rule_state` of a stateful rule (amphione), one tensor per unit, only on the
  unit's owner, saved under `state.<fqn>.rule_state.<unit position>`. Nothing is
  gathered to save it, and a resume on another GPU count or layout finds every
  unit's state by its key.
- `radial_state`, the radial metrics' accumulators (a row per unit). A
  non-expert unit's accumulators advance on its metric owner only, so
  `state_dict()` all-reduces those rows first and every rank saves the owner's.

`t` (the step count) is saved with the param groups. The group keys `rule`,
`args`, `weight_decay`, `momentum`, `nesterov` and `lr` are the run's config and
no checkpoint holds them (for the AdamW groups beside DiSCO too, `lr`): the
container supplies the live values to upstream's flat loader, and the scheduler
sets every group's lr from this run's base lr and schedule at the loaded step.
So a resume may change any of them, in either load order (`dcp.load` loads
`lr_scheduler` before `optimizer`). The owners' fp32 weight copies (the rules
that return the new weight) are not saved: the first step after a load fills
them from the parameters.

A plain resume is bitwise with the uninterrupted run. Limit: a resume cannot
turn averaging ON for a group that kept no buffer (`momentum` 0 -> > 0, or a
group moved off amphione to sgd / muon): the checkpoint has no
`momentum_buffer` for it and `dcp.load` stops with `Missing key in checkpoint
state_dict: optimizer.state.<fqn>.momentum_buffer`. Changing beta while the
group averages, or turning averaging off, works.

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

**Cadence.** The norms log on step 1, every `log_norm_freq`-th step and the
last step (a fixed cadence: there is no dense early phase). Every norm in
`norms_to_log` is computed on every norm-logging step: the SVD-based ones
(`rms_to_rms`, `stable_rank`, `condition_number`, `effective_rank*`, the spectra,
the spectral radial metrics) all read one spectrum per unit.

**Model statistics** (`optimizers/model_stats.py`, off by default, on the
`log_norm_freq` cadence):
- `log_residual_stats`: the RMS of every block's input and residual write and
  of the final norm's input, taken in opt_moe's eager layer loop between the
  compiled blocks (the blocks' graphs are the same either way).
- `log_gain_stats`: each RMSNorm gain's mean, RMS and `||dg|| / ||g||`; with
  column scales (`amphione` `column_scales`) also each
  unit's c (`col_scale_*`: rms, mean log|c|, q10 / q50 / q90, relative update),
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
1.05 (its configs spelled the key `safety`, now an unknown arg).
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
- **Pure DDP with MoE**: not a tested setup (no test runs MoE under pure DDP;
  production runs FSDP).
- **Logging steps** pay a blocking device-to-host copy in the MoE publish,
  which exposes DiSCO's kernel launches after it (~70 ms of GPU time measured
  on mis-8b at FSDP 16).
- **Normalised rules and idle rows**: a row whose gradient stops keeps taking
  full-size steps along its decaying momentum for ~300 steps (rare embedding
  rows, idle experts). Measured, loss effect unmeasured.
- **PP**: supported by the optimizer (a group that matches nothing on a stage
  is dropped there), but without the LM-head geometry.
- **Gram rules on sharded units** (`compute: gram_polar` on a sharded head or
  embedding) issue a blocking all-reduce behind the async all-to-all, so the
  expert compute waits for the exchange. Not the production rules.
- **W&B resume / fork** (`WANDB_RESUME_FROM`, `WANDB_FORK_FROM`) get no rank
  suffix under per-rank logging.
- The metrics gather sends to every rank; only the logging rank reads it.
- **Weights changed outside `step()`** (weight surgery, a heal or outer step
  that writes the parameters only) desynchronise the owners' fp32 weight copies.
  The container compares them with the parameters on every norm-logging step and
  every 100 steps (`_COPY_CHECK_EVERY`) and raises on every rank; between checks
  the rule runs on the stale copy. Writes that exist today are safe: the averages'
  start and validation swap (before step 1 / restored bitwise), EMA and the MoE
  bias hook (no parameter writes). A new writer must rebuild the copies
  (`load_state_dict` does) or run before the first step.
