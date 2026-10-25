# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import functools
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torchtitan.config import Configurable

from .optimizer import OptimizersContainer

logger = logging.getLogger(__name__)


__all__ = [
    "LRSchedulersContainer",
]


def _with_group_warmup(
    lr_lambda: Callable, warmup_steps: int, current_step: int
) -> float:
    """The shared schedule times a param group's own linear warmup: the
    optimizer's step t (LambdaLR's current_step t - 1) runs at
    min(1, t / warmup_steps) of it."""
    return lr_lambda(current_step) * min(1.0, (current_step + 1) / warmup_steps)


_DECAY_TYPES = ("linear", "sqrt", "cosine")


def _with_group_decay(
    lr_lambda: functools.partial,
    warmup_steps: int,
    decay_ratio: float | None,
    decay_type: str | None = None,
    min_lr_factor: float | None = None,
) -> functools.partial:
    """The shared schedule with a param group's own `decay_ratio` (the decay
    covers that share of the total steps, at most every step after the
    group's warmup), `decay_type` and `min_lr_factor` (the floor of its decay).
    A key left None keeps the shared schedule's."""
    kw = dict(lr_lambda.keywords)
    if decay_ratio is not None:
        total = kw["warmup_steps"] + kw["stable_steps"] + kw["decay_steps"] - 1
        decay = min(
            round(total * decay_ratio), total - max(warmup_steps, kw["warmup_steps"])
        )
        kw.update(
            stable_steps=total + 1 - kw["warmup_steps"] - decay, decay_steps=decay
        )
    if decay_type is not None:
        kw["lr_decay_type"] = decay_type
    if min_lr_factor is not None:
        kw["min_lr_factor"] = min_lr_factor
    return functools.partial(lr_lambda.func, **kw)


def _check_table(table) -> tuple[tuple[int, float], ...]:
    try:
        knots = tuple((int(s), float(v)) for s, v in table)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"group lr_table needs (step, factor) knots: {ex}") from ex
    if len(knots) < 2:
        raise ValueError("group lr_table needs at least two knots")
    if any(b[0] <= a[0] for a, b in zip(knots, knots[1:])):
        raise ValueError("group lr_table steps must increase")
    if not all(math.isfinite(v) and v >= 0 for _, v in knots):
        raise ValueError("group lr_table factors must be finite and >= 0")
    return knots


def _table_factor(knots: tuple[tuple[int, float], ...], current_step: int) -> float:
    """The table at the optimizer's step t = current_step + 1: linear between
    knots, the end values outside them."""
    t = current_step + 1
    i = bisect.bisect_right(knots, (t, math.inf))
    if i == 0:
        return knots[0][1]
    if i == len(knots):
        return knots[-1][1]
    (s0, v0), (s1, v1) = knots[i - 1], knots[i]
    return v0 if s0 == t else v0 + (v1 - v0) * (t - s0) / (s1 - s0)


def _group_lambdas(optimizer, lr_lambda: Callable) -> Callable | list[Callable]:
    """The shared schedule, or one per param group when a group sets its own
    `warmup_steps`, `decay_ratio`, `decay_type` or `min_lr_factor` (extra keys
    in its optimizer kwargs), or its own `lr_table` (knots (step, factor) that
    replace the schedule for that group; not with the other keys)."""
    tables = [g.get("lr_table") for g in optimizer.param_groups]
    for g, table in zip(optimizer.param_groups, tables):
        if table is not None:
            mixed = [
                k
                for k in ("warmup_steps", "decay_ratio", "decay_type", "min_lr_factor")
                if g.get(k) is not None
            ]
            if mixed:
                raise ValueError(
                    f"group lr_table replaces the schedule: not with {mixed}"
                )
    knots = [None if t is None else _check_table(t) for t in tables]
    if any(k is not None for k in knots):
        rest = _group_lambdas_no_table(optimizer, lr_lambda)
        rest = rest if isinstance(rest, list) else [rest] * len(knots)
        return [
            lam if k is None else functools.partial(_table_factor, k)
            for lam, k in zip(rest, knots)
        ]
    return _group_lambdas_no_table(optimizer, lr_lambda)


def _group_lambdas_no_table(
    optimizer, lr_lambda: Callable
) -> Callable | list[Callable]:
    own = [
        (
            int(g.get("warmup_steps", 0)),
            g.get("decay_ratio"),
            g.get("decay_type"),
            g.get("min_lr_factor"),
        )
        for g in optimizer.param_groups
    ]
    for _, _, t, _ in own:
        if t is not None and t not in _DECAY_TYPES:
            raise ValueError(f"group decay_type {t!r}: one of {_DECAY_TYPES}")
    if not any(
        w or d is not None or t is not None or m is not None for w, d, t, m in own
    ):
        return lr_lambda
    out = []
    for w, d, t, m in own:
        if d is None and t is None and m is None:
            lam = lr_lambda
        else:
            lam = _with_group_decay(
                lr_lambda,
                w,
                None if d is None else float(d),
                t,
                None if m is None else float(m),
            )
        out.append(functools.partial(_with_group_warmup, lam, w) if w else lam)
    return out


class LRSchedulersContainer(Stateful, Configurable):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Note**
    Users who want to customize the lr_scheduler behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same
    signature as ``torch.optim.lr_scheduler.LRScheduler`` class: ``step()``, ``state_dict()``,
    ``load_state_dict()``.

    **Checkpoint behavior**
    All schedulers share the same lambda and step together. On save, only
    ``last_epoch`` is saved. On load, ``last_epoch`` is restored and each
    scheduler recomputes its lr from its own optimizer's ``base_lrs``. This
    handles mixed optimizers (different base lrs) and resharding (different
    number of schedulers between save and load).

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        warmup_steps: int = 200
        """
        Steps for lr scheduler warmup, normally 1/5 of --training.steps
        """

        total_steps: int | None = None
        """
        Total steps for LR schedule calculation. If None, defaults to training.steps.
        This allows decoupling the LR schedule from the actual training steps,
        which is useful for debugging with fewer steps while maintaining the same LR curve,
        or for early stopping scenarios.
        """

        decay_ratio: float | None = None
        """
        Controls the proportion of the training steps allocated to the learning rate decay phase.
        If `None`, the learning rate will begin decaying immediately after the warmup period.
        Otherwise, the learning rate will remain stable after the warmup period and
        only start decaying during the last `decay_ratio` portion of the total training steps.
        This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
        """

        decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
        """
        Learning rate decay type to use during training:
        - 'linear': linearly decays learning rate from initial to final value
        - 'sqrt': decays learning rate following a 1 minus square root curve
        - 'cosine': smoothly decays learning rate following a cosine curve
        """

        min_lr_factor: float = 0.0
        """
        Min lr ratio for lr scheduler.
        If provided, the range of decay factor is scaled from 1 to `min_lr_factor`
        to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.min_lr_factor`.
        """

        # pyrefly: ignore [bad-override]
        def build(self, *, optimizers, training_steps):
            """Build a LRSchedulersContainer from this config.

            Args:
                optimizers: The corresponding OptimizersContainer.
                training_steps: The total number of training steps.

            Returns:
                A LRSchedulersContainer for the given optimizers.
            """
            # Use total_steps from config if set, otherwise fall back to training_steps
            total_steps = (
                self.total_steps if self.total_steps is not None else training_steps
            )

            warmup_steps = int(self.warmup_steps)

            if warmup_steps > total_steps:
                logger.warning(
                    f"Warmup steps ({warmup_steps}) exceed total steps ({total_steps}). "
                    f"Adjusting warmup steps to {total_steps}."
                )
                warmup_steps = total_steps

            if self.decay_ratio is not None:
                decay_steps = round(total_steps * self.decay_ratio)
                if warmup_steps + decay_steps > total_steps:
                    logger.warning(
                        f"Warmup ({warmup_steps}) + decay ({decay_steps}) steps exceed "
                        f"total steps ({total_steps}). "
                        f"Adjusting decay steps to {total_steps - warmup_steps}."
                    )
                    decay_steps = total_steps - warmup_steps
            else:
                decay_steps = total_steps - warmup_steps
            # Add a virtual last step to prevent the learning rate from dropping to 0
            stable_steps = total_steps + 1 - warmup_steps - decay_steps
            lr_decay_type = self.decay_type
            min_lr_factor = self.min_lr_factor

            def linear_warmup_stable_decay(
                current_step: int,
                warmup_steps: int,
                stable_steps: int,
                decay_steps: int,
                lr_decay_type: str,
                min_lr_factor: float,
            ):
                """
                Computes linear warmup followed by stable learning rate for a while,
                then some type of decay.

                Per LambdaLR requirement, this is accomplished by returning
                a multiplicative factor `curr_adjustment` ranging from 1 to 0
                to adjust the learning rate to create the desired schedule.

                We offer three types of learning rate decay schedules:
                1. `linear`: decays linearly from 1 to 0 over the decay period.
                2. `sqrt`: decays as 1 minus the square root of the decay progress.
                3. `cosine`: follows a cosine curve, decaying according to the values of the half-period of the cosine function.

                If `min_lr_factor` is specified, the decay range is scaled from 1 to `min_lr_factor`
                to ensure the learning rate does not drop below this minimum value.
                """
                warmup_stable_steps = warmup_steps + stable_steps
                if current_step < warmup_steps:
                    # linear warmup
                    # 0-indexed step, hence + 1 adjustments
                    current_step += 1
                    assert (
                        warmup_steps != 0
                    ), "warmup_steps must not be zero to reach this branch"
                    curr_adjustment = float(current_step / warmup_steps)
                elif current_step < warmup_stable_steps:
                    curr_adjustment = 1.0
                else:
                    # 0-indexed step, hence + 1 adjustments
                    current_step += 1
                    assert (
                        decay_steps != 0
                    ), "decay_steps must not be zero to reach this branch"
                    progress = float(current_step - warmup_stable_steps) / decay_steps

                    if lr_decay_type == "linear":
                        curr_adjustment = 1 - progress
                    elif lr_decay_type == "sqrt":
                        curr_adjustment = 1 - math.sqrt(progress)
                    elif lr_decay_type == "cosine":
                        curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
                    curr_adjustment = (
                        min_lr_factor + (1 - min_lr_factor) * curr_adjustment
                    )
                return curr_adjustment

            lr_lambda = functools.partial(
                linear_warmup_stable_decay,
                warmup_steps=warmup_steps,
                stable_steps=stable_steps,
                decay_steps=decay_steps,
                lr_decay_type=lr_decay_type,
                min_lr_factor=min_lr_factor,
            )
            return LRSchedulersContainer(optimizers, lr_lambda)

    schedulers: list[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert (
            len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [
            LambdaLR(optimizer, _group_lambdas(optimizer, lr_lambda))
            for optimizer in optimizers
        ]

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def get_metrics(self) -> dict[str, float]:
        """Return learning rates keyed by optimizer (and param-group index)."""
        metrics = {}
        for scheduler in self.schedulers:
            opt_name = type(scheduler.optimizer).__name__
            last_lrs = scheduler.get_last_lr()
            for i, lr_val in enumerate(last_lrs):
                key = f"lr/{opt_name}" if len(last_lrs) == 1 else f"lr/{opt_name}/{i}"
                metrics[key] = float(lr_val)
        return metrics

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # Only last_epoch is needed — each scheduler recomputes its lr from
        # its own optimizer's base_lrs on load. Per-scheduler state (base_lrs,
        # _last_lr) is not saved because it's reconstructed from the optimizer
        # config at construction time.
        return {"last_epoch": self.schedulers[0].last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Only restore last_epoch. Each scheduler recomputes _last_lr from its
        # own optimizer's base_lrs and the shared lambda. This is correct for
        # mixed optimizers (different base_lrs) and resharding (different number
        # of schedulers between save and load).
        #
        # NOTE: torchtitan's LR schedules are stateless functions
        # of (last_epoch, base_lr) — LambdaLR with a pure lambda. If a stateful
        # scheduler (e.g. ReduceLROnPlateau) is added, this method must be updated
        # to restore additional state.
        last_epoch = state_dict["last_epoch"]
        for scheduler in self.schedulers:
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = scheduler.get_lr()
            # The next step runs this run's schedule, not the lr the checkpoint
            # holds (a branch's own table); on a plain resume they are equal.
            for group, lr in zip(scheduler.optimizer.param_groups, scheduler._last_lr):
                group["lr"] = lr
