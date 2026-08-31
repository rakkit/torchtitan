# Validation and Evaluation

`torchtitan` provides direct and indirect support for validation to support user's training goals. Direct support is provided by the `Validator` class which interacts directly with the training loop, and indirect support is provided through [HuggingFace checkpoint conversion](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md#huggingface) for users who want to do evaluation using external tools such as ELeutherAI's `lm_eval`.

## Validation
For users who want to perform validation directly during the training loop, we provide the `Validator` class which can be conveniently configured via `Validator.Config` in your config_registry function. The validator class has access to and reuses many of the trainer's functions such as its parallelization, including pipelining.

Below is an example validation config:

```python
validator=Validator.Config(
    freq=500,
    dataset="c4_validation",
    steps=-1,  # consumes the entire validation set
),
```

## Third-Party Evaluation
With `./scripts/checkpoint_conversion/convert_to_hf.py`, `torchtitan` offers support for converting checkpoints from DCP to safetensors format. Using this script, users can perform efficient evaluation separate from their training using external libraries that support HuggingFace e.g. `lm_eval` with `vllm` backend.

## Offline DCP validation loss

`scripts/evaluate_checkpoint.py` evaluates one native DCP checkpoint without
starting the training loop. It loads the model only, evaluates every named set
in a Python configuration file, and writes one CSV per set. This is useful when
an external Bash or Slurm script is sweeping many training checkpoints.

The evaluator uses DDP only (`dp_replicate=WORLD_SIZE`, no FSDP/TP/PP/CP or
EP) and is intended for dense models that fit replicated on one node. Launch
one rank per visible GPU; single-node runs with 1--8 GPUs are supported.
It reuses the newest `job_config_*.json` snapshot in the training run directory
to recover the model architecture, tokenizer, parameter dtype, and AMP setup.
Pass `--job-config` to select a different snapshot explicitly.

Create an evaluation config such as `configs/validations.py`:

```python
from torchtitan.evaluation import EvaluationDatasetConfig, EvaluationSuiteConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader


def make_evaluation_suite() -> EvaluationSuiteConfig:
    return EvaluationSuiteConfig(
        output_dir="/path/to/offline-validation",
        validation_sets=[
            EvaluationDatasetConfig(
                name="val-a",
                dataloader=HuggingFaceTextDataLoader.Config(
                    dataset=["simple_custom"],
                    dataset_path=["/path/to/val-a"],
                    dataset_split=["train"],
                    dataset_key=["text"],
                    dataset_streaming=True,
                    infinite=False,
                    pack_strategy="greedy",
                    num_workers=1,
                    pin_memory=True,
                ),
                # Omit these two fields to inherit them from job_config.
                seq_len=4096,
                local_batch_size=8,
            ),
            # Add more named validation sets here. The loaded model is reused.
        ],
    )
```

Launch it with the same distributed environment that you use for training:

```bash
torchrun --standalone --nproc-per-node="${NUM_GPUS}" \
    scripts/evaluate_checkpoint.py \
    --dump-folder /path/to/training-run \
    --step 5000 \
    --eval-config /path/to/configs/validations.py
```

Set `NUM_GPUS` to the number of GPUs allocated to the single node (between 1
and 8). It must match the number of GPUs visible to the process, for example
after Slurm sets `CUDA_VISIBLE_DEVICES`.

Pass `--max-batches` and/or `--local-batch-size` to override every named set's
own `max_batches`/`local_batch_size` for that invocation — useful for a quick
smoke test against a large corpus without editing the config file, e.g.
`--max-batches 1 --local-batch-size 1` to exercise the whole pipeline on a
single batch. Omit either flag to use the config's own value (falling back to
the checkpoint's training `local_batch_size` and to evaluating every batch,
respectively).

For each set, rank 0 upserts a row in `<output_dir>/<name>.csv`, keyed by the
absolute checkpoint path and step. The row contains `total_nll`,
`total_tokens`, `total_bytes`, nats/token, BPB, PPL, and elapsed time. The CSV
write is serialized with a lock file, so independent Slurm jobs can safely
write different checkpoint rows to the same result file.

BPB is calculated from UTF-8 bytes attached to the original raw-text character
span of every scored target token. The evaluator gets these spans from the
tokenizer's offset mapping, then carries their byte lengths through packing.
Only `greedy` packing is supported, which preserves the same document-spanning
behavior as training. At each sequence boundary, the first token is input
context rather than a scored label, so its raw bytes are also excluded from
BPB; `total_nll`, `total_tokens`, and `total_bytes` always refer to the same
scored targets. The evaluator requires a finite, single-source dataset and
rejects tokenizers without reliable character offsets, unsupported packing
modes, and unsupported MoE/parallelism layouts rather than emitting a
misleading metric.

### Example usage of `lm_eval` with `vllm`:
To use this specific setup make sure to include a HuggingFace `config.json` file which is not provided by conversion script or `last_save_in_hf` option. The HF config file can be downloaded by running `python ./scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets config`.

Note that pip installing `lm-eval` may result in breaking `torchtitan` dev environment so we recommend creating a separate env.
```bash
pip install "lm-eval[vllm]"
lm_eval --model vllm \
    --model_args pretrained=./outputs/checkpoint/step-1000,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8, \
    --tasks mmlu \
    --batch_size auto
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6209|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5481|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.7045|±  |0.0078|
| - social sciences|      2|none  |      |acc   |↑  |0.7351|±  |0.0078|
| - stem           |      2|none  |      |acc   |↑  |0.5357|±  |0.0085|
