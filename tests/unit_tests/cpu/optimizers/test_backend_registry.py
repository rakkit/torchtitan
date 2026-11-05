# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every zeropower backend a run can name must exist, whatever loaded first.

`gram_polar_express` lives in `kernels/gram_newton_schulz.py`, which imports
`kernels/zeropower.py` for the polar-express coefficients, so whichever of the
two a process reaches first leaves the other partially initialised. That once
cost a production config its backend: the registration sat in `zeropower`
behind a bare `except Exception: pass`, and a new import elsewhere silently
reversed the order. A subprocess per order is the only honest check -- within
one interpreter the first import wins for the whole session.
"""

import subprocess
import sys

import pytest

from torchtitan.optimizers.disco.rules import rule_of
from torchtitan.optimizers.kernels.zeropower import zeropower_backends

# The backends a shipped recipe or the template config may name.
PRODUCTION_BACKENDS = ["newtonschulz5", "polar_express", "gram_polar_express", "svd"]

_PROBE = """
import importlib
importlib.import_module({first!r})
from torchtitan.optimizers.kernels.zeropower import zeropower_backends
print(",".join(sorted(zeropower_backends)))
"""


@pytest.mark.parametrize("backend", PRODUCTION_BACKENDS)
def test_backend_resolves_through_a_param_group(backend):
    r = rule_of({"rule": "muon", "args": {"compute": backend, "steps": 5}})
    assert r.compute.backend == backend


@pytest.mark.parametrize(
    "first",
    [
        "torchtitan.optimizers",  # what a training run does
        "torchtitan.optimizers.kernels.zeropower",
        "torchtitan.optimizers.kernels.gram_newton_schulz",
        "torchtitan.optimizers.disco.rules",
        "torchtitan.optimizers.disco.presets",
    ],
)
def test_registry_is_complete_whatever_is_imported_first(first):
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(first=first)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = set(out.stdout.strip().splitlines()[-1].split(","))
    assert set(PRODUCTION_BACKENDS) <= got, sorted(set(PRODUCTION_BACKENDS) - got)


def test_registration_is_not_silenced():
    """The bug was invisible because the import that registers it was wrapped
    in a bare `except`. Keep the registration unconditional."""
    src = open(
        zeropower_backends["gram_polar_express"].__module__.replace(".", "/") + ".py"
    ).read()
    assert 'zeropower_backends["gram_polar_express"]' in src


_PRESETS_PROBE = """
import importlib
importlib.import_module({first!r})
from torchtitan.optimizers.disco.rules import RULES
print(",".join(sorted(RULES)))
"""


@pytest.mark.parametrize(
    "first",
    ["torchtitan.optimizers.disco.rules", "torchtitan.optimizers.disco.presets"],
)
def test_the_built_in_presets_register_whatever_is_imported_first(first):
    """`rules.py` holds the registry and imports `presets.py` at its end, which
    registers the built-ins: either module first must give all of them."""
    out = subprocess.run(
        [sys.executable, "-c", _PRESETS_PROBE.format(first=first)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = set(out.stdout.strip().splitlines()[-1].split(","))
    assert got == {"sgd", "muon", "hyperball", "amphione"}
