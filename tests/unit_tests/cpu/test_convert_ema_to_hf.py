# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.optimizer import EMA

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "checkpoint_conversion"
    / "convert_ema_to_hf.py"
)

# Distinguishable fills: the trained weights must never reach the export.
TRAINED = 7.0
AVERAGED = -999.0


def _load_script():
    spec = importlib.util.spec_from_file_location("convert_ema_to_hf", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConvertEmaToHf(unittest.TestCase):
    """The export must carry EMA weights for every parameter, including the
    fused ones (QKVLinear's wqkv, FeedForward's w13) whose state-dict keys are
    split by save hooks and therefore differ from the EMA's own FQNs.
    """

    model_name = "llama3"
    model_flavor = "debugmodel"
    buffer_patterns: list[str] = []
    # Buffers get their own sentinels so master-vs-EMA can be told apart in the
    # export. buffer_name is filled whether or not buffer_patterns tracks it.
    buffer_name = "expert_bias_E"
    buffer_trained = 3.0
    buffer_averaged = -55.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.module = _load_script()

    def _write_checkpoint(self):
        """Save a checkpoint laid out the way CheckpointManager writes one:
        model tensors flattened at the top level, EMA nested under "ema"."""
        spec = importlib.import_module(
            f"torchtitan.models.{self.model_name}"
        ).model_registry(self.model_flavor)
        with torch.device("cpu"):
            model = spec.model.build()
        ema = EMA.Config(buffer_patterns=self.buffer_patterns).build(
            model_parts=[model]
        )
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(TRAINED)
            for name, buf in model.named_buffers():
                if self.buffer_name in name:
                    buf.fill_(self.buffer_trained)
        for ema_opt in ema._param_optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"].fill_(AVERAGED)
        for ema_opt in ema._buffer_optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"].fill_(self.buffer_averaged)

        ckpt = str(Path(self.tmp) / "step-1")
        states = {"ema": ema}
        states.update(ModelWrapper(model).state_dict())
        dcp.save(states, checkpoint_id=ckpt)
        return ckpt, model

    def _convert_and_capture(self, ckpt):
        """Run the real conversion, intercepting only the HF write."""
        captured = {}

        def fake_save(state_dict, **kwargs):
            captured["sd"] = state_dict

        with mock.patch.object(self.module.dcp, "save", fake_save), mock.patch.object(
            self.module, "HuggingFaceStorageWriter", mock.MagicMock()
        ):
            self.module.convert_ema_to_hf(
                Path(ckpt),
                Path(self.tmp) / "hf_out",
                self.model_name,
                self.model_flavor,
                None,
                "float32",
            )
        self.assertIn("sd", captured, "conversion never reached the HF write")
        return captured["sd"]

    def test_export_carries_ema_weights_for_every_parameter(self):
        ckpt, _ = self._write_checkpoint()
        exported = self._convert_and_capture(ckpt)

        self.assertTrue(exported, "nothing was exported")
        trained_keys = []
        averaged_keys = []
        live_buffer_keys = []
        for key, value in exported.items():
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            first = float(value.flatten()[0])
            if first == TRAINED:
                trained_keys.append(key)
            elif first == AVERAGED:
                averaged_keys.append(key)
            elif self.buffer_patterns and first == self.buffer_trained:
                live_buffer_keys.append(key)

        self.assertEqual(
            trained_keys,
            [],
            f"{len(trained_keys)} exported tensors still hold the trained "
            f"weights instead of the EMA, e.g. {sorted(trained_keys)[:4]}",
        )
        self.assertTrue(averaged_keys, "no exported tensor holds the EMA value")
        self.assertEqual(
            live_buffer_keys,
            [],
            f"EMA-tracked buffers exported their live value: {live_buffer_keys}",
        )

    def test_end_to_end_write_puts_ema_weights_in_the_safetensors(self):
        """The other tests intercept the HF write. This one runs the whole
        conversion and reads the safetensors back off disk, so the feature's
        actual output is checked rather than the dict handed to the writer."""
        from safetensors.torch import load_file

        ckpt, _ = self._write_checkpoint()
        out = Path(self.tmp) / "hf_e2e"
        self.module.convert_ema_to_hf(
            Path(ckpt), out, self.model_name, self.model_flavor, None, "float32"
        )
        shards = sorted(out.glob("*.safetensors"))
        self.assertTrue(shards, f"no safetensors written into {out}")

        trained, averaged = [], []
        for shard in shards:
            for key, value in load_file(str(shard)).items():
                if value.numel() == 0:
                    continue
                first = float(value.flatten()[0])
                if first == TRAINED:
                    trained.append(key)
                elif first == AVERAGED:
                    averaged.append(key)
        self.assertEqual(
            trained, [], f"trained weights reached disk: {sorted(trained)[:4]}"
        )
        self.assertTrue(averaged, "no EMA weights on disk")

    def test_export_dtype_is_applied_without_losing_the_ema(self):
        """--export_dtype casts after the tt->hf conversion, so it has to be
        checked against the written file, not the dict."""
        from safetensors.torch import load_file

        ckpt, _ = self._write_checkpoint()
        for export_dtype, torch_dtype in (
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ):
            with self.subTest(export_dtype=export_dtype):
                out = Path(self.tmp) / f"hf_{export_dtype}"
                self.module.convert_ema_to_hf(
                    Path(ckpt),
                    out,
                    self.model_name,
                    self.model_flavor,
                    None,
                    export_dtype,
                )
                tensors = {
                    key: value
                    for shard in sorted(out.glob("*.safetensors"))
                    for key, value in load_file(str(shard)).items()
                }
                self.assertTrue(tensors)
                self.assertEqual({v.dtype for v in tensors.values()}, {torch_dtype})
                # the sentinel, rounded into the export dtype
                want = torch.tensor(AVERAGED, dtype=torch_dtype).item()
                dont = torch.tensor(TRAINED, dtype=torch_dtype).item()
                firsts = {float(v.flatten()[0]) for v in tensors.values() if v.numel()}
                self.assertNotIn(dont, firsts)
                self.assertIn(want, firsts)

    def test_fused_parameters_are_actually_exercised(self):
        """Guards the guard: if the flavor stopped using fused parameters, the
        test above would pass even against the original bug."""
        spec = importlib.import_module(
            f"torchtitan.models.{self.model_name}"
        ).model_registry(self.model_flavor)
        with torch.device("meta"):
            model = spec.model.build()
        param_fqns = {name for name, _ in model.named_parameters()}
        sd_keys = set(model.state_dict())
        fused = param_fqns - sd_keys
        self.assertTrue(
            fused,
            "no fused parameters in this flavor -- the export test can no "
            "longer detect the fused/split key mismatch",
        )

    def test_checkpoint_without_ema_is_rejected(self):
        spec = importlib.import_module(
            f"torchtitan.models.{self.model_name}"
        ).model_registry(self.model_flavor)
        with torch.device("cpu"):
            model = spec.model.build()
        ckpt = str(Path(self.tmp) / "no-ema")
        dcp.save(dict(ModelWrapper(model).state_dict()), checkpoint_id=ckpt)
        with self.assertRaises(ValueError) as caught:
            self.module.convert_ema_to_hf(
                Path(ckpt),
                Path(self.tmp) / "hf_out2",
                self.model_name,
                self.model_flavor,
                None,
                "float32",
            )
        self.assertIn("no EMA weights", str(caught.exception))


class TestConvertEmaToHfMoEWithBufferEma(TestConvertEmaToHf):
    """A different family and a different shape of EMA state: deepseek_v3 has
    MoE expert_bias_E buffers, which EMA tracks alongside the parameters."""

    model_name = "deepseek_v3"
    model_flavor = "debugmodel"
    buffer_patterns = [r"expert_bias_E$"]


class TestConvertEmaToHfUntrackedBuffers(TestConvertEmaToHf):
    """The mixed case: EMA covers the parameters, but a buffer that EMA does
    not track has to be exported from the master weights rather than dropped.
    deepseek_v3 has real expert_bias_E buffers, and buffer_patterns is empty
    here, so none of them are averaged.
    """

    model_name = "deepseek_v3"
    model_flavor = "debugmodel"
    buffer_patterns: list[str] = []

    def test_untracked_buffers_come_from_the_master_weights(self):
        from safetensors.torch import load_file

        ckpt, model = self._write_checkpoint()
        buffers = {
            name for name, _ in model.named_buffers() if self.buffer_name in name
        }
        self.assertTrue(buffers, f"this flavor has no {self.buffer_name} buffers")

        out = Path(self.tmp) / "hf_untracked"
        self.module.convert_ema_to_hf(
            Path(ckpt), out, self.model_name, self.model_flavor, None, "float32"
        )
        tensors = {
            key: value
            for shard in sorted(out.glob("*.safetensors"))
            for key, value in load_file(str(shard)).items()
        }
        self.assertTrue(tensors)

        from_master = [
            key
            for key, value in tensors.items()
            if value.numel() and float(value.flatten()[0]) == self.buffer_trained
        ]
        from_ema = [
            key
            for key, value in tensors.items()
            if value.numel() and float(value.flatten()[0]) == AVERAGED
        ]
        leaked_params = [
            key
            for key, value in tensors.items()
            if value.numel() and float(value.flatten()[0]) == TRAINED
        ]
        # untracked buffers present, carrying the master value
        self.assertEqual(
            len(from_master),
            len(buffers),
            f"expected {len(buffers)} untracked buffers from the master weights, "
            f"got {len(from_master)}",
        )
        # parameters still averaged, and no master parameter leaked
        self.assertTrue(from_ema)
        self.assertEqual(leaked_params, [])


class TestConvertEmaToHfFrozenParameters(unittest.TestCase):
    """A run with frozen parameters has no EMA state for them. They still have
    to reach the export, from the trained weights -- and for a *fused* frozen
    parameter that only works by pushing the loaded weights back through the
    module, because the state dict's fused entries do not alias the live
    parameters.
    """

    model_name = "llama3"
    model_flavor = "debugmodel"
    frozen_fqn = "layers.0.attention.qkv_linear.wqkv.weight"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.module = _load_script()

    def test_frozen_fused_parameter_exports_the_trained_weights(self):
        from safetensors.torch import load_file

        spec = importlib.import_module(
            f"torchtitan.models.{self.model_name}"
        ).model_registry(self.model_flavor)
        with torch.device("cpu"):
            model = spec.model.build()
        params = dict(model.named_parameters())
        self.assertIn(self.frozen_fqn, params, "this flavor has no fused QKV")
        params[self.frozen_fqn].requires_grad_(False)

        with torch.no_grad():
            for param in model.parameters():
                param.fill_(TRAINED)
        ema = EMA.Config().build(model_parts=[model])
        self.assertNotIn(
            self.frozen_fqn,
            set(ema.optimizers[0].param_groups[0]["param_names"]),
            "a frozen parameter must not be EMA-tracked",
        )
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"].fill_(AVERAGED)

        ckpt = str(Path(self.tmp) / "step-1")
        states = {"ema": ema}
        states.update(ModelWrapper(model).state_dict())
        dcp.save(states, checkpoint_id=ckpt)

        out = Path(self.tmp) / "hf"
        self.module.convert_ema_to_hf(
            Path(ckpt), out, self.model_name, self.model_flavor, None, "float32"
        )
        tensors = {
            key: value
            for shard in sorted(out.glob("*.safetensors"))
            for key, value in load_file(str(shard)).items()
        }
        self.assertTrue(tensors)

        # the frozen fused parameter's split keys must hold the trained value
        frozen_keys = [
            key
            for key in tensors
            if "layers.0.self_attn" in key and "_proj" in key and "o_proj" not in key
        ]
        self.assertEqual(len(frozen_keys), 3, f"expected q/k/v, got {frozen_keys}")
        for key in frozen_keys:
            self.assertEqual(
                float(tensors[key].flatten()[0]),
                TRAINED,
                f"{key} did not come from the trained weights",
            )

        # and nothing anywhere may still hold a freshly-initialised value
        stray = [
            key
            for key, value in tensors.items()
            if value.numel() and float(value.flatten()[0]) not in (TRAINED, AVERAGED)
        ]
        self.assertEqual(
            stray,
            [],
            f"{len(stray)} tensors hold neither the EMA nor the trained value "
            f"(uninitialised), e.g. {sorted(stray)[:3]}",
        )


if __name__ == "__main__":
    unittest.main()
