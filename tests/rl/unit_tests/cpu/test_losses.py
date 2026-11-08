# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.trainer.losses.dapo import _normalize, DAPOLoss


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = _normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


def test_dapo_loss_matches_vllm_tempered_logprobs_bitwise() -> None:
    torch.manual_seed(0)
    logits = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 128, (64,))
    temperature = torch.full((64,), 0.7)
    # vLLM's sampler: logits.to(float32).div_(temperature), then log_softmax(dtype=float32).
    generator_logprobs = (
        logits.to(torch.float32)
        .div_(temperature[:, None])
        .log_softmax(dim=-1, dtype=torch.float32)
        .gather(-1, labels[:, None])
        .squeeze(-1)
    )

    _, metrics = DAPOLoss.Config().build()(
        logits=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def test_dapo_truncated_is_gradient_is_capped_ratio_times_advantage() -> None:
    """TIS (truncated_is_cap): each token's gradient is min(p_trainer / p_generator, cap)
    * advantage * grad log p_trainer / N (from the equation, not the code), and with
    the generator equal to the trainer it is DAPO's gradient."""
    torch.manual_seed(0)
    logits = torch.randn(8, 16, dtype=torch.float64)
    labels = torch.randint(0, 16, (8,))
    trainer_logprobs = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    shift = torch.tensor(
        [-1.5, -0.3, 0.0, 0.4, 1.0, -0.9, 0.2, 0.0], dtype=torch.float64
    )
    advantages = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
    cap = 2.0
    kwargs = dict(
        labels=labels,
        global_loss_token_counts=torch.tensor(8),
        temperature=torch.ones(8, dtype=torch.float64),
        advantages=advantages,
        loss_mask=torch.ones(8, dtype=torch.bool),
    )
    pred = logits.clone().requires_grad_(True)
    loss, metrics = DAPOLoss.Config(truncated_is_cap=cap).build()(
        logits=pred, generator_logprobs=trainer_logprobs + shift, **kwargs
    )
    loss.backward()
    ref = logits.clone().requires_grad_(True)
    weight = torch.clamp(torch.exp(-shift), max=cap)
    logp = ref.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    (-(weight * advantages * logp).sum() / 8).backward()
    torch.testing.assert_close(pred.grad, ref.grad)
    assert metrics["loss/tis_capped_frac"] == 2 / 8  # exp(1.5), exp(0.9) > 2

    grads = []
    for config in (DAPOLoss.Config(truncated_is_cap=cap), DAPOLoss.Config()):
        pred = logits.clone().requires_grad_(True)
        config.build()(logits=pred, generator_logprobs=trainer_logprobs, **kwargs)[
            0
        ].backward()
        grads.append(pred.grad)
    torch.testing.assert_close(grads[0], grads[1])


def test_dapo_mismatch_mask_drops_only_far_tokens_gradient() -> None:
    """IcePop (mismatch_mask_ratio): a token whose trainer / generator ratio is
    outside [1 / beta, beta] gives no gradient; the others keep today's gradient,
    and the denominator is unchanged."""
    torch.manual_seed(0)
    logits = torch.randn(8, 16)
    labels = torch.randint(0, 16, (8,))
    trainer_logprobs = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    # tokens 0 and 1 far from the generator (ratio e^2 and e^-2), the rest equal
    shift = torch.zeros(8)
    shift[0], shift[1] = -2.0, 2.0
    kwargs = dict(
        labels=labels,
        global_loss_token_counts=torch.tensor(8),
        generator_logprobs=trainer_logprobs + shift,
        temperature=torch.ones(8),
        advantages=torch.linspace(-1.0, 1.0, 8),
        loss_mask=torch.ones(8, dtype=torch.bool),
    )
    grads = {}
    for beta in (None, 2.0):
        pred = logits.clone().requires_grad_(True)
        loss, metrics = DAPOLoss.Config(mismatch_mask_ratio=beta).build()(
            logits=pred, **kwargs
        )
        loss.backward()
        grads[beta] = pred.grad
    assert metrics["loss/mismatch_masked_frac"] == 2 / 8
    assert torch.equal(grads[2.0][:2], torch.zeros_like(grads[2.0][:2]))
    assert torch.equal(grads[2.0][2:], grads[None][2:])
    assert grads[None][:2].abs().sum() > 0
