"""The load-balance aux loss must not scale with gradient accumulation.

opt_moe moved from the fork's ``MoEAuxLoss`` (which returned the aux term up
through ``(logits, load_balance_loss)`` and divided by
``gradient_accumulation_steps`` in the trainer) to upstream's ``AuxLoss``
injection path, where ``NormMoE`` injects the gradient at the layer.

``AuxLoss.inject`` expects an UNNORMALISED per-microbatch sum and divides by the
step's global valid-token count.  Our loss functions end in
``loss_per_seq.mean()``, so ``NormMoE`` converts by weighting with the
microbatch's own valid-token count.

Getting that conversion wrong is silent, and both ways were tried during the
migration:

* weight by the GLOBAL denominator -> it cancels ``inject``'s division exactly,
  every microbatch contributes a full ``coeff * mean``, and the aux gradient
  scales with the accumulation count.
* skip the conversion entirely -> the O(1) mean is divided by ~8.4M tokens and
  load balancing is effectively switched off.

These tests pin the arithmetic that sits between those two failures.
"""

import torch

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.opt_moe.norm_moe import LoadBalanceLoss

COEFF = 0.001


def _accumulate(per_microbatch_means, tokens_per_microbatch):
    """Replay NormMoE's conversion over one optimizer step.

    Returns the value ``instance_acc`` holds at the end of the step, which is
    what the per-layer ``moe_load_balance_loss/L-*`` metric reads.
    """
    global_valid = torch.tensor(float(sum(tokens_per_microbatch)))
    layer = LoadBalanceLoss.Config(coeff=COEFF).build()
    for mean, n_valid in zip(per_microbatch_means, tokens_per_microbatch):
        carrier = torch.zeros(1, requires_grad=True)
        # NormMoE: raw_sum = mean * this microbatch's valid tokens.
        raw_sum = torch.tensor(float(mean), requires_grad=True) * float(n_valid)
        layer(raw_sum, carrier=carrier, denominator=global_valid).sum().backward()
    return float(layer.instance_acc)


def test_metric_is_invariant_under_gradient_accumulation():
    """Same tokens and same per-token loss, split 1 way vs 4, must agree."""
    one = _accumulate([0.5], [4096])
    four = _accumulate([0.5] * 4, [1024] * 4)
    assert abs(one - four) < 1e-6, (
        f"aux metric scales with gradient accumulation: gas=1 -> {one}, "
        f"gas=4 -> {four}. Check the raw_sum conversion in NormMoE."
    )


def test_uneven_microbatches_give_the_token_weighted_mean():
    """Packed data makes valid-token counts differ between microbatches.

    A plain average over microbatches would give 0.5 here; the token-weighted
    mean is 0.75, and that is the one that matches how the main loss is
    normalised.
    """
    got = _accumulate([1.0, 0.0], [3072, 1024])
    assert abs(got - 0.75) < 1e-6, f"expected token-weighted 0.75, got {got}"


def test_coefficient_scales_the_gradient_not_the_metric():
    """``coeff`` must not leak into the logged value.

    ``inject`` applies ``coeff`` to the injected gradient but accumulates the
    metric unscaled, so the logged load-balance loss stays comparable across
    ``coeff`` settings -- the property the old straight-through
    ``loss + (aux - aux.detach())`` also had.
    """
    for coeff in (0.001, 0.1):
        layer = LoadBalanceLoss.Config(coeff=coeff).build()
        carrier = torch.zeros(1, requires_grad=True)
        raw_sum = torch.tensor(0.5, requires_grad=True) * 1024.0
        layer(raw_sum, carrier=carrier, denominator=torch.tensor(1024.0)).sum().backward()
        assert abs(float(layer.instance_acc) - 0.5) < 1e-6, (
            f"coeff={coeff} leaked into the logged metric: "
            f"{float(layer.instance_acc)}"
        )
