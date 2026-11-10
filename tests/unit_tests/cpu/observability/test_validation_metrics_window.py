# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.observability import metrics
from torchtitan.observability.logging import NoColor
from torchtitan.observability.metrics import DeviceMemStats, MetricsProcessor


class _FakeMemoryMonitor:
    def __init__(self) -> None:
        self.resets = 0

    def reset_peak_stats(self) -> None:
        self.resets += 1

    def get_peak_stats(self) -> DeviceMemStats:
        return DeviceMemStats(0, 0.0, 0, 0.0, 0, 0)


def test_validation_is_measured_on_its_own_window(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: now[0])
    logged = {}

    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.loggers = [SimpleNamespace(log=lambda m, step: logged.update(m))]
    processor.device_memory_monitor = _FakeMemoryMonitor()
    # 50 s of training (and checkpointing) since the last training log.
    processor.ntokens_since_last_log = 5000
    processor.data_loading_times = [0.5, 0.5]
    processor.time_last_log = 50.0
    processor.step_last_log = 8

    processor.reset()
    processor.ntokens_since_last_log += 800  # what the validator counts
    now[0] = 104.0
    processor.log_validation(loss=1.0, step=10)

    assert logged["validation_metrics/throughput(tps)"] == 200
    assert processor.data_loading_times == []
    assert processor.device_memory_monitor.resets == 1
    assert processor.step_last_log == 10


def _processor(logged: dict, *, logs_global: bool) -> MetricsProcessor:
    processor = object.__new__(MetricsProcessor)
    processor.parallelism_context = SimpleNamespace(non_data_parallel_size=1)
    processor.color = NoColor()
    processor.loggers = [SimpleNamespace(log=lambda m, step: logged.update(m))]
    processor.device_memory_monitor = _FakeMemoryMonitor()
    processor.gpu_peak_flops = 1e15
    processor.has_quantization = False
    processor.ntokens_since_last_log = 5000
    processor.data_loading_times = [0.5]
    processor.time_last_log = 50.0
    processor.step_last_log = 8
    processor._logs_global_metrics = logs_global
    return processor


def test_global_metrics_only_from_the_metrics_rank(monkeypatch) -> None:
    # every rank has a logger (save_for_all_ranks): the loss is the same on all of them, so
    # only the metrics rank logs it; per-GPU and rank-local values come from every rank
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: 60.0)
    for logs_global in (True, False):
        logged: dict = {}
        _processor(logged, logs_global=logs_global).log(
            10,
            2.5,
            3.0,
            0.7,
            {"lr": 1e-3},
            num_flops=1e12,
            local_metrics={"data_docs/a": 4},
        )
        assert ("loss_metrics/global_avg_loss" in logged) is logs_global
        assert ("grad_norm" in logged) is logs_global and (
            "lr" in logged
        ) is logs_global
        assert "throughput(tps)" in logged and logged["data_docs/a"] == 4
        logged.clear()
        _processor(logged, logs_global=logs_global).log_validation(loss=1.0, step=10)
        assert ("validation_metrics/loss" in logged) is logs_global
        assert "validation_metrics/throughput(tps)" in logged
