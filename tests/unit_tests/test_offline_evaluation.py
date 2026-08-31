# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.evaluation.config import (
    EvaluationConfigError,
    find_latest_job_config,
    force_ddp_evaluation_parallelism,
    load_training_config,
    resolve_checkpoint,
    validate_single_node_gpu_world_size,
)
from torchtitan.evaluation.data import (
    build_evaluation_dataloader,
    ByteTrackingGreedyPackedDataset,
    tokenize_document_with_byte_spans,
    TokenizedDocument,
)
from torchtitan.evaluation.runtime import (
    dense_token_nll,
    EvaluationTotals,
    result_row,
    upsert_result,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama3.config_registry import llama3_debugmodel


class DummyTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.eos_id = 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        token_ids = [ord(char) for char in text]
        if add_bos:
            token_ids.insert(0, 1)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def encode_with_offsets(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ):
        token_ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_bos:
            token_ids.insert(0, 1)
            offsets.insert(0, (0, 0))
        if add_eos:
            token_ids.append(self.eos_id)
            offsets.append((len(text), len(text)))
        return token_ids, offsets

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id > 2)

    def get_vocab_size(self) -> int:
        return 512


class TestOfflineEvaluationConfig(unittest.TestCase):
    def test_resolve_checkpoint_requires_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint" / "step-12"
            checkpoint.mkdir(parents=True)
            with self.assertRaisesRegex(EvaluationConfigError, "missing .metadata"):
                resolve_checkpoint(root, 12)
            (checkpoint / ".metadata").touch()
            self.assertEqual(resolve_checkpoint(root, 12), checkpoint.resolve())

    def test_latest_job_config_and_model_rehydration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = llama3_debugmodel()
            original.hf_assets_path = "/tmp/tokenizer-assets"
            original.model_spec.model.dim = 123
            original.training.seq_len = 777

            older = root / "job_config_20260101-0000.json"
            newest = root / "job_config_20260101-0001.json"
            older.write_text(json.dumps(original.to_dict()))
            newest.write_text(json.dumps(original.to_dict()))

            self.assertEqual(find_latest_job_config(root), newest)
            restored = load_training_config(newest)
            self.assertEqual(restored.config.hf_assets_path, "/tmp/tokenizer-assets")
            self.assertEqual(restored.config.training.seq_len, 777)
            self.assertEqual(restored.config.model_spec.model.dim, 123)

    def test_rejects_model_converter_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job_config.json"
            snapshot = llama3_debugmodel().to_dict()
            snapshot["model_converters"] = {"converters": [{"unsupported": True}]}
            path.write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(EvaluationConfigError, "model_converters"):
                load_training_config(path)

    def test_evaluation_parallelism_uses_the_launched_world_size(self):
        config = llama3_debugmodel()
        force_ddp_evaluation_parallelism(config, world_size=3)
        self.assertEqual(config.parallelism.data_parallel_replicate_degree, 3)
        self.assertEqual(config.parallelism.data_parallel_shard_degree, 1)
        self.assertEqual(config.parallelism.tensor_parallel_degree, 1)

    def test_evaluation_parallelism_rejects_unsupported_gpu_counts(self):
        config = llama3_debugmodel()
        with self.assertRaisesRegex(EvaluationConfigError, "1 to 8"):
            force_ddp_evaluation_parallelism(config, world_size=9)

    def test_single_node_gpu_count_must_match_world_size(self):
        validate_single_node_gpu_world_size(
            world_size=3, visible_gpu_count=3, local_world_size=3
        )
        with self.assertRaisesRegex(EvaluationConfigError, "one rank per visible GPU"):
            validate_single_node_gpu_world_size(
                world_size=3, visible_gpu_count=4, local_world_size=3
            )
        with self.assertRaisesRegex(EvaluationConfigError, "single-node only"):
            validate_single_node_gpu_world_size(
                world_size=3, visible_gpu_count=3, local_world_size=2
            )


class TestOfflineEvaluationMetrics(unittest.TestCase):
    def test_tokenizer_offsets_preserve_utf8_byte_spans(self):
        document = tokenize_document_with_byte_spans(DummyTokenizer(), "éa")
        self.assertEqual(document.token_ids.tolist(), [1, ord("é"), ord("a"), 2])
        self.assertEqual(document.token_byte_lengths.tolist(), [0, 2, 1, 0])

    def test_tokenizer_without_offsets_is_rejected(self):
        tokenizer = DummyTokenizer()
        tokenizer.encode_with_offsets = None
        with self.assertRaisesRegex(EvaluationConfigError, "character offsets"):
            tokenize_document_with_byte_spans(tokenizer, "é")

    def test_loader_supports_greedy_packing_with_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "validation.json"
            data_file.write_text('{"text": "é"}\n{"text": "a"}\n')
            config = HuggingFaceTextDataLoader.Config(
                dataset=["simple_custom"],
                dataset_path=[temp_dir],
                dataset_files=["validation.json"],
                dataset_split=["train"],
                dataset_key=["text"],
                infinite=False,
                pack_strategy="greedy",
                num_workers=0,
            )
            dataloader = build_evaluation_dataloader(
                config,
                tokenizer=DummyTokenizer(),
                dp_rank=0,
                dp_world_size=1,
                seq_len=4,
                local_batch_size=1,
            )
            batches = list(dataloader)
            self.assertEqual(
                sum(batch_bytes.sum().item() for _, _, batch_bytes in batches), 3
            )
            self.assertEqual(
                sum((labels != -100).sum().item() for _, labels, _ in batches), 4
            )

    def test_greedy_tracks_only_scored_utf8_target_spans(self):
        # The first token of each greedy window is context, not a scored target.
        # The document's "b" token therefore has no NLL and no BPB denominator.
        document = TokenizedDocument(
            token_ids=np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64),
            token_byte_lengths=np.asarray([0, 2, 1, 1, 1, 0], dtype=np.int64),
        )
        packed = ByteTrackingGreedyPackedDataset(
            [document], seq_len=2, drop_long_samples=False
        )
        samples = list(packed)
        self.assertEqual(len(samples), 2)
        self.assertEqual(sum(batch_bytes.item() for _, _, batch_bytes in samples), 4)
        self.assertEqual(
            sum((labels != -100).sum().item() for _, labels, _ in samples), 4
        )

        logits = torch.zeros(1, 2, 5)
        total_nll = sum(
            dense_token_nll(logits, labels.unsqueeze(0)).item()
            for _, labels, _ in samples
        )
        self.assertAlmostEqual(total_nll, 4 * math.log(5), places=6)

        totals = EvaluationTotals(total_nll=total_nll, total_tokens=4, total_bytes=4)
        self.assertAlmostEqual(totals.loss_nats_per_token, math.log(5), places=6)
        self.assertAlmostEqual(totals.bpb, math.log2(5), places=6)
        self.assertAlmostEqual(totals.ppl, 5.0, places=6)

    def test_csv_upsert_replaces_matching_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "val-a.csv"
            totals = EvaluationTotals(total_nll=8.0, total_tokens=4, total_bytes=2)
            first = result_row(
                checkpoint_path=Path("/checkpoint/step-1"),
                step=1,
                job_config_path=Path("/run/job_config.json"),
                seq_len=128,
                totals=totals,
                elapsed_seconds=1.0,
            )
            upsert_result(csv_path, first)
            replacement = {**first, "total_nll": "9.0"}
            upsert_result(csv_path, replacement)

            lines = csv_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("9.0", lines[1])


class TestOfflineEvaluationCheckpointLoading(unittest.TestCase):
    def test_model_wrapper_loads_only_model_state_from_dcp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = nn.Linear(2, 2)
            expected = {
                key: value.detach().clone()
                for key, value in source.state_dict().items()
            }
            dcp.save(ModelWrapper(source).state_dict(), checkpoint_id=temp_dir)

            target = nn.Linear(2, 2)
            wrapper = ModelWrapper(target)
            state_dict = wrapper.state_dict()
            dcp.load(state_dict, checkpoint_id=temp_dir)
            wrapper.load_state_dict(state_dict)

            for key, value in expected.items():
                self.assertTrue(torch.equal(target.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
