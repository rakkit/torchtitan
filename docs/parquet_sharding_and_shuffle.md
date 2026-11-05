# Parquet sharding and shuffling

How a corpus on disk becomes each rank's token stream, and what actually shuffles it.

This describes `torchtitan/components/data/parquet_manifest.py` and `parquet_stream.py`,
which are fork additions. Everything downstream of them -- mixing, packing, collation --
is upstream's.

## 1. The layout

```
corpus/                       one dataset directory
  manifest.json               built once, read by every rank at startup
  part-00000.parquet          a SHARD
    [row group 0][row group 1][row group 2] ...
  part-00001.parquet
  ...
```

A **row group** is parquet's atomic decode unit: you cannot read one row without
decompressing the group it lives in. Measured on `Nemotron-CC-v2/High-Quality`, a group
is 32,768 rows and decodes to **111.5 MiB**. That single fact drives every decision below.

The **manifest** records, per shard, the number of rows in each row group, plus the
column names and types. It is built from parquet **footers** only -- no data pages are
read -- so building all 45 production manifests (92,160 shards) takes **323 s**.

The manifest describes what is on disk. The training config decides what to train on:

| manifest (a fact about the data) | config (a decision) |
|---|---|
| shard paths, row-group row counts | which column to read (`text_key`) |
| available columns and arrow types | mixing `weight` |
| totals, build provenance, source mtime | `alias`; which datasets are in this stage |

## 2. Rank assignment: row ranges, not files and not row groups

Rank `r` of `W` takes the contiguous global row range `[r*N/W, (r+1)*N/W)` and reads only
the row groups overlapping it. Per-rank row counts differ by at most 1.

The two obvious alternatives both fail on real data:

* **File-granular** (`shard i -> rank i % W`): 28 production corpora have 128 shards and
  68 have 256, so at DP=1024 most ranks would get *nothing*. This is also why HuggingFace
  `datasets` falls back to row-striding, where **every rank opens every file** and
  discards `(W-1)/W` of what it decodes.
* **Row-group-granular**: measured across 40 corpora, **23 of 40 have exactly one row
  group per shard**, so this degenerates to file-granular for most of the data.

Row ranges degrade gracefully instead of failing: where there are many row groups each
rank gets whole distinct ones; where there are few, `ceil(W / num_row_groups)` ranks share
a group, each decompressing it and keeping its slice. `sharing_factor()` reports this and
the source logs a warning when it exceeds 1.

**Balancing is by ROW COUNT, never token count.** A `token_count` column exists in most
corpora but is neither guaranteed present nor guaranteed accurate. Equal rows per rank
gives roughly equal tokens per rank because the corpora are pre-shuffled on disk -- which
is also why no global shuffle is needed to make a rank's slice representative.

## 3. Why streaming, not random access

The decisive number is throughput, not memory. Measured on one real shard, 200 rows at
uniformly random indices with a one-row-group cache:

```
iter_batches, sequential        342,808 rows/s
random access, 1-group cache         11.3 rows/s     (9 of 200 cache hits)
```

A **30,000x** gap. Grain's map path hands a source a *globally shuffled* index, so
consecutive requests land in different row groups and a per-dataset cache almost never
hits; each miss decompresses 111.5 MiB to serve one row.

The memory figure often quoted alongside this -- one decoded group per dataset, 46 x ~124
MiB ~= 5.7 GiB/rank -- is a side effect of the cache, not the problem. On a GH200 each
rank has its own ~120 GB Grace CPU, so 5.7 GiB is ~5% of one rank's RAM and no reason on
its own to avoid random access.

Streaming makes reads sequential by construction, so one group decode is amortised over
its 32,768 rows. Resident set is one batch:

```
read_row_group(0)        111.5 MiB
iter_batches(bs=1024)      4.5 MiB
iter_batches(bs=4096)     14.2 MiB
```

## 4. Shuffling

A rank's row range is cut into **spans** -- the pieces of row groups it overlaps. Two
mechanisms operate on spans, and they are orthogonal.

### `reshuffle_spans_per_epoch` (default on) -- decorrelates EPOCHS

Permutes the order of the rank's spans, seeded by `(dataset_seed, epoch)`. It shuffles a
list of tens-to-hundreds of items, so it is free, and the permutation is *recomputed* from
`(seed, epoch)` on resume rather than stored.

Epoch 0 is permuted too. Exempting it made the first pass -- the only pass a short run has
-- the one with the weakest mixing.

### `num_concurrent_spans` (default 8) -- breaks up LOCAL adjacency

Keeps N spans open at once and draws each row from one of them. Because the N open spans
come from a *permuted* list they sit far apart on disk, so consecutive output rows come
from N widely separated places in the corpus. Reads stay strictly sequential *within* each
span; the cost is N decode buffers (~4.5 MiB each), not N row groups.

Seeking to an arbitrary row group is a footer lookup at a byte offset, not a scan --
measured flat at 4-9 ms whether you start at row group 0 or 23 -- which is what makes
span-granular shuffling affordable at all.

### Why not a sliding window

`shuffle=True` inserts grain's `WindowShuffleIterDataset`. A window of size W can only move
a row W positions, so with sequential input it reshuffles *locally* and epoch 2 replays
epoch 1. Measured on a 12,800-row / 64-span corpus ("disk span" = how much of the corpus a
512-row stretch of OUTPUT covers; "epoch rho" = order correlation between consecutive
epochs, 0 = independent, mean of 3 seeds):

```
configuration                        disk span   #spans   epoch rho
window 1024 only                          8.0%      6.1   +0.983
span permutation only                    61.1%      3.5   -0.062
span perm + window 1024                  78.4%      6.1   -0.080
span perm + 8 concurrent spans           82.0%     10.2   -0.041
span perm + 8 spans + window 1024        85.4%     12.7   -0.061
```

The window alone is the bad option on both axes. Span permutation is what takes epoch
correlation from +0.98 to ~0; concurrent spans are what widen the output's disk coverage.

### Cost

Measured on the real corpus at DP=4, median of 3 x 30k rows after warmup:

```
sequential, no permutation             168,173 rows/s   1.00x
span permutation, width 1              161,455 rows/s   0.96x
span permutation, width 8 (default)    165,331 rows/s   0.98x
span permutation, width 32             227,921 rows/s   1.36x
span permutation, width 8 + window 10k  85,390 rows/s   0.51x
```

Peak RSS growth was 9 MiB for every non-window case. Training consumes ~115k tokens/s per
rank, about 130 rows/s, so even the slowest option supplies ~650x more than is consumed.
Shuffling is free; only the row-window has a real cost, and it buys nothing the other two
do not.

## 5. Behaviour as DP grows

Once there are more ranks than row groups, a rank's whole share fits inside one row group
and it gets a single span. Both mechanisms then become no-ops -- permuting a list of one,
and a width capped at `min(8, 1)` -- so every epoch replays the identical order.

`split_spans_for_interleave` prevents this by subdividing the rank's row range. A span is
a `(shard, row_group, start, stop)` slice, so this is arithmetic and opens no extra file.
The cost is that sibling sub-spans of one row group each decode that group's prefix to
reach their slice; a 64-row floor stops it degenerating into per-row seeking.

Measured on the 44-corpus production mixture, as it is on disk today:

```
    DP   min spans/rank   after split   worst sharing   degenerate?
  1024                3             7               1   no
  2048                2             5               1   no
  4096                1             5               2   no
  8192                1             3               4   no
```

The fewest row groups in that mixture is 2,048 (`Nemotron-DQA`), so the raw slicing only
reaches one span per rank at DP >= 4096. **No re-splitting of the parquet is needed**; the
code-side split covers it. What smaller row groups would buy is a lower sharing factor at
DP >= 4096, which is read amplification, not a shuffle or correctness problem.

## 6. Checkpoint state

Per dataset, per rank, plain integers -- inspectable and diffable:

```python
{"epoch": 2,
 "slots": [{"span_index": 5, "row_in_span": 1801}, ...],   # one per concurrent span
 "next_span": 13,
 "draw_index": 90211,
 "next_index": 412903,          # documents emitted, CUMULATIVE across epochs
 "spans_fingerprint": [26, 840552]}
```

* The span permutation and the draw sequence are pure functions of `(seed, epoch[, block])`
  and are recomputed on resume, never stored. Draws are generated in blocks of 8192 so a
  resume recomputes one block instead of replaying millions.
* `next_index` is cumulative so `data_docs/{alias}` is monotone; counting within an epoch
  made it saw-tooth to zero whenever a small corpus wrapped.
* `spans_fingerprint` is `(span count, row count)`. `span_index` indexes a permutation of
  *this* span list, so if the corpus gained shards and the manifest was rebuilt the same
  integers name different rows. `check_freshness` cannot catch that -- rebuilding is the
  documented remedy and re-records the directory mtime -- so a mismatched fingerprint is a
  hard error rather than a silently shifted traversal.

## 7. Knobs

| knob | default | when to change it |
|---|---|---|
| `num_concurrent_spans` | 8 | Lower to 1 for byte-reproducible disk order. Higher costs one decode buffer each. |
| `reshuffle_spans_per_epoch` | True | Off gives identical order every epoch; only for reproduction. |
| `batch_size` | 1024 | The resident set per dataset (~4.5 MiB). |
| `shuffle` (loader) | see below | Adds the row window on top. Rarely worth it; see section 4. |
| `read_in_order` (factory) | False | **Set True for every validator.** Sets the two above to 1/False so a validation set scores the same held-out tokens on every run. |

`GrainDataLoader.Config.shuffle` defaults to `True` upstream. Both fork factories override
it to `False`: the corpora are pre-shuffled, and `True` also renames the checkpoint state
key from `parent` to `parent_window_start_state`, which silently zeroed every
`data_docs/{alias}` series until `cursor_position` was taught to raise instead of
returning 0.
