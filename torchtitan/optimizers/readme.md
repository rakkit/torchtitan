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
  utils.py          FQN cleanup, the DiSCO kwargs the container passes
  disco/
    optimizer.py    DiSCO: build (families, exchanges, metric registry), step(), state
    layout.py       families (classify), per-rank row tables
    rules.py        what a rule is: Phase, Compute, Chain, Rule, Result; the registry
    presets.py      the built-ins: pre phases, computes, presets sgd/muon/right_polar/
                    lm_head/left_polar/aus
    transport.py    Exchange (one packed all-to-all pair), Replicated (all-gather)
    metrics.py      MetricSink: per-unit registry, one flat buffer, one gather, unpack
  kernels/          orthogonaliser backends: zeropower, triton Newton-Schulz,
                    Gram Newton-Schulz, underflow floors
  helpers/          norm_helper, gram_helper, radial_helper, head_geometry, power_iteration
  export/           spectrum / gram-vector Parquet export (W&B artifacts)
models/common/param_layout.py   the declaration vocabulary models use (ParamLayout, UnitSpec)
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
  AdamW beside DiSCO), `lr`, `weight_decay`, `momentum`, `nesterov`, `rule`,
  `args`. Every key the config file sets is a CLI flag:
  `--optimizer.groups.default.lr 0.05`, `--optimizer.groups.default.args.steps 8`.
- `args` merge over the default group's. A group with the default's `rule`
  takes all of the default's args; a group with another rule takes only the
  ones its preset accepts, minus the preset's `own` ones (below). So a group
  whose rule differs must state every arg that should not come from the
  default: under a `muon` default with `compute: gram_polar_express`, a bare
  `{"rule": "sgd"}` group inherits that orthogonaliser.
- Build errors: unknown keys; the renamed ones (`backend`, `norm_factor`,
  `pre_norm`, which name their replacement); DiSCO-only keys (`rule`, `args`,
  `momentum`, `nesterov`) on another optimizer's group; a DiSCO with no default
  `lr`; `gram_level` outside 0-3; an iterative compute with `steps < 1`; the
  vocabulary matrices under a gathering rule; a Gram compute on a wide unit;
  declared units that overlap or misstate their shape; a stateful rule whose
  state shape differs across one parameter's units.
- `moe_template_config()`'s groups are a valid run on their own: Muon on the
  matrices, the row rules on the embedding, head and routers (by kind).
- `optimizer.groups` and upstream's `optimizer.param_groups` are alternatives;
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
| `expert_batch` | dim 0 is a batch of independent matrices, one unit each |
| `embedding`, `unembedding` | the vocabulary matrices, rows are tokens |
| `router` | an MoE router's gate, rows are experts |
| `vector`, `scalar` | a one-row unit `[1, d]` / `[1, 1]` |

opt_moe's modules declare what only they know (expert batches, fused q/k/v and
w1/w3, the router gate), finding each parameter by identity (`named`); the
model adds the two vocabulary matrices and states the rest by rank (`by_rank`,
which refuses rank 3+: a depthwise Conv1d `[C, 1, K]` and an expert batch
`[E, F, D]` look alike). A fused parameter's units are unions of contiguous
row runs, so they get unfused math and log under the split names
(`...wqkv` + `wq` logs as `...wq`).

Checked at build: an undeclared parameter; a vocabulary matrix under a rule
that would gather all its rows on one rank; a Gram compute on a wide unit
(rank-deficient Gram); and the ranks of a model part compare a hash of their
plans, so a divergent plan raises instead of hanging the first exchange.

## Rules: pre -> compute -> post -> scale

A rule is a small optimizer over one unit, run as a pipeline; the optimizer
derives every collective from what its pieces declare, and a rule never sees a
transport.

| stage | config (`args`) | choices |
|---|---|---|
| pre | `pre` | `identity`, `row-l2`, `col-l2`, `mat-l2`, `col-center`. A phase acts on the rows a rank holds, or declares `reduce` and supplies a fixed-size statistic summed over every rank holding rows of the unit (one packed all-reduce for all units). Phases are told the unit's true row count, never a shard's |
| compute | `compute` (+ `steps`) | the only place a direction is formed. Row maps `identity`, `row-l2`, `sign`; orthogonalisers `newtonschulz5`, `newtonschulz5_triton`, `polar_express`, `polar_express_triton`, `gram_polar_express`, `svd`; `gram_polar` (the polar factor through the `[n, n]` Gram); chains `main+row_map+...` (`gram_polar+row-l2`) |
| post | (preset-internal) | phases that need a statistic over all rows after the compute (the LM head's re-centring) |
| scale | `scale` | the unit's step relative to `lr`, from its shape `[m, n]` = `[d_out, d_in]`: `one`, `spectral` (sqrt(m/n)), `image_spectral`, `n`, `sqrt_n`, `inv_n`, `inv_sqrt_n`, `sqrt_mn`. A constant per unit, the last multiply of the compiled compute (for a Gram compute, or after a reducing post phase, it runs after it) |

A `Compute` declares where the unit must be whole (`kind`: `rows` = the rows a
rank holds, `gram` = those rows plus one `[n, n]` all-reduce, `unit` = whole on
its owner), whether it needs the pre-update weight (`needs_weight`), whether it
keeps state (`has_state`, shape `state_shape(m, n)`) and what it returns
(`output`). One apply serves every output: `"update"` ->
`w <- (1 - lr wd) w - lr U`; `"update_coeffs"` -> per unit `w <- alpha w - beta U`
(alpha and beta ride one small all-reduce); `"weight"` -> `w <- W_next` (the step
`W - W_next` travels in the update's slot). The apply adds lr and weight decay
only to a plain update: a coefficient or weight output must fold them in
itself, from the `ctx` it is given (`StepCtx(lr, wd, t)`), and cannot have
post phases or a scale. A contract that cannot hold fails when the rule is
built, not at the first step.

Presets, each a function registered with `@register(name, own=...)` whose
keyword arguments are its args and defaults:

| `rule` | pipeline | defaults | `own` |
|---|---|---|---|
| `sgd` | `pre` -> `compute` -> scale | identity, identity, steps 0, `one` | |
| `muon` | `pre` -> orthogonaliser -> scale | identity, `newtonschulz5`, 5, `spectral` | |
| `right_polar` | `pre` -> `gram_polar` -> scale; no all-to-all | 5 steps, `spectral`, work dtype fp16 | `compute` |
| `lm_head` | centre -> `gram_polar+row-l2` (`spectral: false`: `row-l2`) -> centre -> scale | 5 steps, `inv_sqrt_n` | `scale` |
| `left_polar` | centre -> orthogonaliser -> centre -> scale (routers) | `newtonschulz5`, 5, `spectral` | |
| `aus` | `AUSMap` around a muon direction: needs the weight, keeps a per-unit vector, returns `beta U` and `alpha` | as `muon`, `sigma_iters` 8, `eta`: an `EtaSchedule` dict (`kappa`, `tau`, `alpha`, `min`, `anneal_*`; default `0.1 / sqrt(t)`) or `"lr"` | |

`right_polar`, `lm_head` and `left_polar` are the symmetry-compatible rules of
Lau & Su (arXiv 2605.18106); the paper's other two are configs: row-norm only
is `{"rule": "sgd", "args": {"pre": "row-l2", "compute": "identity", "scale":
"one"}}` (stated in full, since an `sgd` group under a `muon` default inherits
its compute and scale), the SwiGLU hybrid `{"rule": "muon", "args": {"pre":
"row-l2"}}`. `aus` over a row compute stays that row rule; over a Gram compute
it is a build error. The production row rules are
`compute: row-l2` with `scale: sqrt_n` (embedding) or `inv_sqrt_n` (head,
routers). `lm_head` and `left_polar` take no `pre`: the centring is the rule.

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

Owners are `unit.index % world` after a big-to-small sort (snaked over owner
blocks for the replicated family). Per-peer row tables come from DTensor's own
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
with the param groups; `rule` and `args` are not: a resumed run follows its own
config, and the container supplies them to upstream's flat loader. Each expert
block's moments are views of one buffer; `state_dict()` hands out copies (DCP's
save staging cannot deep-copy a DTensor over an offset view) and a load copies
them back. A resume is bitwise with the uninterrupted run, AUS included.
Flipping `fuse_qkv` / `fuse_w13` changes the unit count, so such a checkpoint
does not resume.

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
