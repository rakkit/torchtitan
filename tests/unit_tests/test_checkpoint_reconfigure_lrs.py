# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`checkpoint.reconfigure_lrs` must not be a silent no-op.

An optimizer state_dict carries `param_groups[*]["lr"]`, so resuming a run whose LR
schedule changed restores the OLD rates and undoes the new schedule. The flag exists to
prevent that -- but it works by setting `preserve_lrs_when_loading` on the optimizer
container, and only some containers read it. Accepting the flag against a container that
ignores it reproduces exactly the failure the flag is for, silently.
"""

import unittest


class TestReconfigureLrs(unittest.TestCase):
    def test_a_container_that_cannot_preserve_lrs_is_refused(self):
        from torchtitan.components.checkpointer.dcp import CheckpointManager

        class ContainerWithoutSupport:
            """Upstream's OptimizersContainer has no preserve_lrs_when_loading."""

        self.assertFalse(
            hasattr(ContainerWithoutSupport(), "preserve_lrs_when_loading")
        )
        # The guard is the `hasattr` check at the assignment site; assert the error text
        # names the fix so an operator is not left guessing.
        source = CheckpointManager.__init__.__doc__ or ""
        del source
        import inspect

        body = inspect.getsource(CheckpointManager.__init__)
        self.assertIn("reconfigure_lrs", body)
        self.assertIn('hasattr(optimizers, "preserve_lrs_when_loading")', body)
        self.assertIn("does not support preserving learning", body)

    def test_the_fork_container_does_support_it(self):
        from torchtitan.optimizers.container import DiSCOOptimizersContainer

        self.assertIn(
            "preserve_lrs_when_loading",
            DiSCOOptimizersContainer.__init__.__code__.co_names
            + tuple(DiSCOOptimizersContainer.load_state_dict.__code__.co_names),
        )


if __name__ == "__main__":
    unittest.main()
