# Datasets

TorchTitan uses one Grain-based data pipeline for text pretraining, SFT, and image training. Mental model:

```text
1. Define a source (e.g. jsonl):
    class: SourceConfig
    output: RandomAccessDataSource | IterDataset

2. Define a dataset (filter/process applied to the source):
    class: SingleDatasetConfig
    does:   pre-filter -> process -> post-filter
    output: MapDataset | IterDataset

3. Compose datasets (optional):
    class: e.g. FirstFitPackingConfig(dataset=DatasetMixConfig(...))
    input:  one or more child DatasetConfig values
    does:   mix, concatenate, and/or pack
    output: MapDataset | IterDataset

4. Dataloader:
    runtime: GrainDataLoader
    config:  GrainDataLoader.Config
    input:  MapDataset | IterDataset
    does:   convert to iterable if needed -> batch -> collate -> prefetch
    output: TrainingMicrobatch

5. Trainer:
    input: TrainingMicrobatch
    does:  model forward and backward
```

# Text Pretraining

## Local JSONL

Each non-empty line must contain one JSON object:

```json
{"title": "First", "body": "The first document."}
{"title": "Second", "body": "The second document."}
```

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import TextProcessor


def article_text(row):
    return row["title"] + "\n\n" + row["body"]


books_ds = SingleDatasetConfig(
    source=IndexedJsonlSource.Config(
        patterns=(
            "/datasets/books/*.jsonl",
        ),
    ),
    processor=TextProcessor.Config(text_fn=article_text),
    post_filters=(lambda sample: sample is not None,),
)

books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=books_packed_ds,
)
```

## Hugging Face sources

Use random access for a materialized dataset:

```python
from torchtitan.components.data import HuggingFaceRandomAccessSource

source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    name="main",
    split="train",
)
```

Use streaming when the corpus should not be materialized:

```python
from torchtitan.components.data import HuggingFaceStreamingSource

source = HuggingFaceStreamingSource.Config(
    path="allenai/c4",
    name="en",
    split="train",
)
```

### Hub rate limits on multi-node jobs

Streaming `allenai/c4` from the Hub happens at trainer init: every rank calls
`datasets.load_dataset`. A large job (for example 48 nodes) can hit Hugging Face
HTTP 429.

Download the dataset once onto shared storage, then point `path` at that
directory. `HuggingFaceStreamingSource.Config.path` and
`HuggingFaceRandomAccessSource.Config.path` accept a Hub id or a local directory
that `datasets.load_dataset` accepts.

Recipes use the `en` config. Download that subset (plus `README.md`, which
declares the config) rather than the full multilingual dump.

```bash
huggingface-cli download allenai/c4 --repo-type dataset --include "en/*" --include "README.md" --local-dir /datasets/c4
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    allow_patterns=["en/*", "README.md"],
    local_dir="/datasets/c4",
)
```

Recipes such as `llama3_8b` use `ConcatThenSplitPackingConfig(dataset=DATASETS["c4"])`.
Replace the source path, or construct the source directly:

```python
from dataclasses import replace

from torchtitan.components.data import HuggingFaceStreamingSource
from torchtitan.components.data.text_datasets import DATASETS

c4 = DATASETS["c4"]
c4 = replace(c4, source=replace(c4.source, path="/datasets/c4"))

source = HuggingFaceStreamingSource.Config(
    path="/datasets/c4",
    name="en",
    split="train",
)
```

If the corpus is fully materialized, use `HuggingFaceRandomAccessSource` with the
same `path`, `name`, and `split`.

Optionally export `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for authenticated Hub
quota. A token does not remove the need to pre-download on large clusters.

Debug recipes already use `DATASETS["c4_test"]` (local JSON under
`tests/assets/c4_test/`) and do not hit the Hub.

## Adding your own source -- Example: Pretokenized data

This example uses token offsets to expose one pretokenized document per index, then reuses the normal processing and packing configs.

```python
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    DatasetIterationPolicy,
    RandomAccessDataSource,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.config import Configurable


class PretokenizedMemmapSource(Configurable, RandomAccessDataSource):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(self, config: Config):
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokensToTextSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, token_ids, rng):
        del rng
        if len(token_ids) < 2:
            return None
        return TextSequence(
            input_ids=token_ids[:-1],
            labels=token_ids[1:],
        )


token_documents_ds = SingleDatasetConfig(
    source=PretokenizedMemmapSource.Config(
        tokens_path="tokens.bin",
        document_offsets_path="document_offsets.npy",
    ),
    processor=TokensToTextSequence.Config(),
    post_filters=(lambda sample: sample is not None,),
)

packed_tokens_ds = ConcatThenSplitPackingConfig(
    dataset=token_documents_ds,
)
```

# SFT

SFT changes the processor and packing policy, not the loader:

```python
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor


def gsm8k_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


gsm8k_ds = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="openai/gsm8k",
        name="main",
        split="train",
    ),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)

gsm8k_packed_ds = FirstFitPackingConfig(dataset=gsm8k_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=gsm8k_packed_ds,
)
```

Without a renderer, `ChatProcessor` applies the tokenizer's chat template to a single-turn
`[user, assistant]` pair, creates next-token input and label pairs, and sets
prompt labels to `IGNORE_INDEX`. It locates the prompt/response boundary by
rendering the prompt with `add_generation_prompt=True` and requiring that to be
an exact token prefix of the full render, raising a `ValueError` when it is not.
Templates that rewrite earlier turns, or turn separators that only merge in
context, break that assumption.

For multi-turn conversations, select the model's renderer explicitly:

```python
from renderers import Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers

processor = ChatProcessor.Config(
    messages_fn=lambda row: row["messages"],
    renderer=from_renderers(Qwen3RendererConfig()),
)
```

The renderer uses TorchTitan's loaded tokenizer and returns tokens with a loss
mask in one pass. The mask supervises model-generated tokens, including turn
terminators, and excludes prompt tokens and template scaffolding. Conversations
must end with an assistant message. Each conversation is one sample; packing
resets positions between conversations, not between turns. Samples exceeding
`max_context_length` are dropped whole.

Formatting and reasoning retention follow the selected renderer. For example,
Qwen3 omits reasoning from assistant turns before the last user query. Those
omitted tokens receive no loss. To train on each turn's reasoning, prepare
separate conversation prefixes in the source dataset. `thinking_retention`
controls the renderer's rollout bridging, not the full renders used here.

# Mixing datasets

Keep each weight next to its dataset:

```python
from torchtitan.components.data import DatasetMixConfig, WeightedDataset

# `books_ds` and `code_ds` are SingleDatasetConfig values.
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.75),
        WeightedDataset(dataset=code_ds, weight=0.25),
    ),
)
```

Examples use probabilities that sum to `1.0`, but any positive relative weights are accepted and normalized internally. `weight=0.75` next to `weight=0.25` means the first dataset is drawn three times as often.

### Mix documents before packing when weights should count documents

```python
packed_pretraining_ds = ConcatThenSplitPackingConfig(
    dataset=pretraining_mix_ds,
)
```

### Pack each child before mixing when weights should count fixed-length rows

```python
books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)
code_packed_ds = ConcatThenSplitPackingConfig(dataset=code_ds)

token_ratio_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_packed_ds, weight=0.67),
        WeightedDataset(dataset=code_packed_ds, weight=0.33),
    ),
)
```

Automatic adjustment from observed document or supervised-token counts is not implemented. A custom mix can maintain per-dataset moving averages and rebalance its weights.

### Weights as token shares per step: the two pretraining loaders

`make_pretrain_dataloader_config(datasets, packing=...)` (`mix.py`) builds the pretraining and midtraining loader from
`DatasetSpec` records. Each weight is that dataset's share of the TOKENS of every optimizer step. One word switches
between the two loaders:

| `packing=` | name | how it mixes | per step, off the weights (4.19M tokens / step, 46 datasets) |
| --- | --- | --- | --- |
| `"concat_then_split_per_dataset"` | per-dataset rows | each dataset packed into its own rows of `max_context_length`; grain's mix picks rows; DP rank r starts at r x rows per microbatch | 3.2% median, 6.5% max (100 steps: 0.6%) |
| `"concat_then_split"` | pool | `TokenWeightedMixConfig`: documents cut into pieces of at most `max_context_length`; the next piece comes from the dataset furthest below its token share; pieces packed into microbatch-long rows | every rank within one piece per dataset |

Example, `max_context_length` 16k, 64k tokens per GPU, web documents A (5k), B (3k), C (40k):

```
per-dataset rows: web's stream cut every 16k, a row = one dataset
  web rows: [A 5k | B 3k | C 0-8k] [C 8-24k] [C 24-40k ...]
  GPU 0:    [web row][code row][math row][web row]
  C's segments: 8k, 8k, 8k, 8k, 8k   (cut at C's own 16k marks AND at every row end)

pool: pieces cut at each document's own 16k marks, mixed by token share, then packed
  web pieces: A 5k, B 3k, C 0-16k, C 16-32k, C 32-40k
  GPU 0:    [A][code][C 0-16k][math][B][qa][C 16-32k ...]   (cut once more at the 64k end)
  C's segments: 16k, 16k, 8k
```

Attention never crosses a position reset (document start, piece start, row start), so a token's context is its
segment. The pool keeps more segments at full length (test fixture, 13-token documents, `max_context_length` 8:
70 full segments of 331 vs 16 of 414 for per-dataset rows), at the price of more global-attention work per token.

State and metrics:
- per-dataset rows: the children's cursors and grain's mix index (each rank's start position, kept across a stage
  switch and a DP change).
- pool: the children's cursors (at the start of a partly emitted document), the next piece per child (`pieces`), and the
  tokens drawn per child (`tokens`). An exact resume re-reads at most one document per child. A stage switch or a DP
  change starts the counts and pieces at 0.
- both log `data_docs/{alias}` (documents read, per rank); the pool also logs `data_tokens/{alias}` (tokens drawn, per
  rank, cumulative): the per-step share of a dataset is the difference between two logged steps summed over ranks.

Cost of the pool: one comparison over the datasets per piece and one partly emitted document per dataset in memory.
`first_fit` uses the same token-weighted mix. SFT (`sft.py`) keeps grain's mix: there a weight selects conversations.

### Concatenate datasets

Use concatenation to treat finite datasets as one corpus. Every row appears once; dataset sizes determine their proportions. Use mixing instead for explicit weights or streaming datasets.

```python
from torchtitan.components.data import DatasetConcatConfig

pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds, code_ds, math_ds),
)
```

To include every row from one finite dataset multiple times per epoch, repeat that child in the concatenation:

```python
pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds,) * 3 + (code_ds, math_ds),
)
```

Every `books_ds` row now appears three times in the combined finite index space. With `shuffle=True`, TorchTitan shuffles that combined index space before DP sharding.

Each child is built with seed `seed + index`, the same as mixed children, so repeated copies make independent draws in random processors (for example augmentation or random cropping). Inserting or reordering a child therefore changes the draws of every later child.

Use mixing instead when the desired behavior is relative source-selection frequency rather than exact finite duplication:

```python
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.6),
        WeightedDataset(dataset=code_ds, weight=0.2),
        WeightedDataset(dataset=math_ds, weight=0.2),
    ),
)
```

`GrainDataLoader.Config(repeat=True)` repeats the complete concatenated or mixed dataset; it does not change one child's relative contribution.

# Images and multimodal data

Images use the same source, dataset, loader, sharding, and checkpoint contracts. Their processors preserve modality-specific sample dictionaries, and their collators create model-specific batches.

Image and video preprocessing is fixed by the model, like the tokenizer: set `Trainer.Config.vision_processor` from the model's settings next to its special tokens, e.g. `VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)`. Its `resize` rule sizes images and its `video_resize` rule sizes video frames (None: the model takes no video). The trainer builds it once and hands it to the dataloader. The dataset's sample processor decodes, resizes and normalizes each image with it (`resize_and_normalize`, or `resize_and_normalize_video` for decoded frames); the collator cuts the microbatch's images and videos into patches with it (`patchify`).

Example for Qwen multimodal:

```python
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import (
    MMSamplePackingConfig,
    VisionSampleProcessor,
    _process_cc12_wd_sample,
)
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS

mm_processor = VisionSampleProcessor.Config(
    sample_processor=_process_cc12_wd_sample,
)

mm_ds = SingleDatasetConfig(
    source=HuggingFaceStreamingSource.Config(
        path="pixparse/cc12m-wds",
        split="train",
    ),
    processor=mm_processor,
    post_filters=(lambda sample: sample is not None,),
)

packed_mm_ds = MMSamplePackingConfig(
    dataset=mm_ds,
    num_packing_bins=8,
)

config.vision_processor = VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)
config.dataloader = GrainDataLoader.Config(
    dataset=packed_mm_ds,
    collator=MultiModalCollator.Config(build_mrope_positions=True),
    streaming_shuffle_buffer_size=128,
)
```

`num_packing_bins` is the number of candidate packed rows kept open, not an input sample buffer.

Custom image augmentation belongs in a `SampleProcessor`.

# Loader policy

Configure run-wide behavior once:

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining_ds,
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_buffer_size=1_000,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=500,
    ),
    num_prefetch_microbatches=2,
)
```

### Reading indexed datasets

A random-access source becomes a `MapDataset`, which supports `dataset[index]`. It becomes an `IterDataset` when a later stage needs to consume samples sequentially:

```text
all children are MapDataset: DatasetMixConfig remains a MapDataset
any child is IterDataset:    DatasetMixConfig converts each MapDataset child
packing:                     converts its child if needed and returns IterDataset
GrainDataLoader:             converts a MapDataset if no earlier stage did
```

`read_options` controls each `MapDataset`-to-`IterDataset` conversion:

```python
grain.ReadOptions(
    num_threads=16,            # indexed samples read concurrently
    prefetch_buffer_size=500,  # samples waiting for the consumer
)
```

Each conversion has its own threads and buffer. An all-map mix converts once; a mix containing a stream converts each map child separately.

### Streaming shuffle and ready batches

`streaming_shuffle_buffer_size` is the number of raw rows retained for approximate shuffling. A larger buffer improves mixing but uses more memory.

`num_prefetch_microbatches` is the number of complete, collated microbatches allowed to wait for the trainer:

```text
trainer computes batch 10
background thread prepares batches 11 and 12
```

# Distributed and checkpoint behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers               -> same rows for their effective-DP coordinate
```

Data ownership is decided before batching:

```text
random-access source -> global shuffle -> contiguous balanced DP shard
Hugging Face stream -> source-level DP shard
DatasetMixConfig    -> combines children already owned by this DP rank
DatasetConcatConfig -> concatenates, globally shuffles, then DP-shards
packing             -> packs samples locally on each DP rank
GrainDataLoader     -> batches and collates that rank's samples
```

For random-access training, each rank receives a contiguous slice of the globally shuffled index space, not a contiguous region of the original corpus.

With effective DP greater than one, `repeat=False` is rejected because ranks can exhaust at different steps and hang training collectives. Use `repeat=True` and let the trainer's step count stop training.

`GrainDataLoader.state_dict()` records:

- source cursors and shuffle/repeat progress
- mix-child and packing-buffer state
- batching and prefetch state
- effective DP degree

Resume requires unchanged code, config, source contents, tokenizer, and effective DP degree.

The exception is `KeyedMixDataLoader` (`make_pretrain_dataloader_config`, parquet
stream sources): it resumes on another effective DP degree, e.g. a later stage with
more HSDP replicas. The checkpointer sees the saved rank items in the checkpoint
metadata and every new rank loads all of them (`request_saved_ranks`); per dataset,
the rows each old rank had not read in its current pass are re-dealt over the new
ranks in contiguous, balanced chunks (`parquet_stream.redealt_state`):

```text
3 old ranks, 63 rows, mid-pass:   r0 read 5 / 21, r1 0 / 21, r2 13 / 21  -> 44 unread
resume on 2 ranks:                new r0 reads 22 of them, new r1 the other 22
then                              each new rank's own share of the next passes
```

Every row of the pass is read exactly once. The packer's partial document and the
mix's draw counter of the old ranks are dropped (the mix restarts with this rank's
fresh state); same-degree resumes read only their own item and stay exact.

### Node-local attention balancing

Each DP rank packs its own documents, so the attention work of equally long
microbatches differs: a full-attention layer costs ~L^2 / 2 per document of
length L, a sliding-window layer ~L * W. Every collective waits for the rank
that drew the long documents. `Trainer.Config.attention_balancer =
NodeAttentionBalancer.Config()` (`balance.py`; off by default) regroups each
microbatch's document segments over the ranks of one node:

```text
one node, 4 ranks, 64 tokens each; documents written as count x length;
cost = full-attention query-key pairs, sum of L (L + 1) / 2
            rank 0                rank 1          rank 2               rank 3
packed      2 x 32       1056     16 x 4   160    8 x 8        288     32 x 2   96
balanced    1 x 32 +              1 x 32 +        4 x 8 +              4 x 8 +
            16 x 2        576     16 x 2   576    8 x 4        224     8 x 4    224
-> the slowest rank: 1056 -> 576
```

1. the node's ranks all-gather their microbatches (token ids, labels,
   positions, padding mask) over Gloo;
2. each rank runs the same deterministic assignment: segments by descending
   cost, each to the least-loaded rank with room for it whole; a segment that
   fits nowhere whole is cut over the ranks with room left;
3. positions restart at 0 in every segment; the loss and routing token counts
   are recomputed for the new rows.

The step's documents, labels and loss are unchanged; only which GPU computes
which document changes. The balancer keeps no state (resume is unaffected).
TP, CP and PP above 1 are refused. Measured (opt_moe l40, 64k tokens per GPU,
32k context, same nodes): +2.0% tokens/s at 16 nodes, +3.8% at 32 nodes.

With equal counts, a rank holding one long document stays the slowest whatever
else it keeps (at 64k context, opt_moe l44: no gain). `NodeComputeBalancer`
lets the token counts differ: a segment costs its attention pairs plus
`token_cost` per token (GEMMs, norms and routing scale with tokens), so that
rank keeps fewer short documents and the others take them, each at most
`max_tokens_ratio` x the row length (activation memory grows with tokens):

```text
4 ranks x 64k tokens, 64k context, one 46k-token document (k = 1024; opt_moe l44,
step 30 of a 16-node run; ms = the recipe's fitted cost model)
equal counts     46k doc + 18k short  5913 ms | 64k  5028 | 64k  5020 | 64k  5017
variable counts  46k doc +  9k short  5245 ms | 67k  5245 | 67k  5245 | 67k  5244
```

`token_cost` is the recipe's measured ratio, from one profiled step: (compute
time - attention time) / tokens over attention time / query-key pairs (pairs
summed over the layers, as `segment_attention_cost` counts them). A FLOP count
misses remat and glue (it gives ~0.56 x the measured ratio for opt_moe l44,
which costs about a quarter of the gain); a wrong value balances less, it
never changes what is trained. The loss is normalized by the global token
count, so variable rows leave it unchanged; compiled blocks run with a dynamic
token dimension (one more graph per block type).

Measured (opt_moe l44, 16 nodes, 64k tokens per GPU, same nodes, step 20):

```text
context   equal counts   variable counts   change   reserved memory
32k       12,602         12,841            +1.9%    77.9 -> 86.7 GiB
16k       13,284         13,184            -0.8%    77.8 -> 83.0 GiB
```

At 32k the slowest rank's compute drops 5,654 -> 5,507 ms (traces of every
rank); what is left is per-rank expert load and noise, which no regrouping of
documents reaches, and variable rows cost about 1% by themselves. Use it where
long documents dominate (32k context and longer), not at 16k.
