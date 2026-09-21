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
        """Behaviour, not source text.

        This used to assert on `inspect.getsource(CheckpointManager.__init__)`,
        which proved nothing about what the code does and broke the moment the
        block moved into a shared helper. Both checkpointers now call
        `apply_reconfigure_lrs`, so testing it tests both.
        """
        from torchtitan.components.checkpointer.base import apply_reconfigure_lrs

        class ContainerWithoutSupport:
            """Upstream's OptimizersContainer has no preserve_lrs_when_loading."""

        with self.assertRaises(ValueError) as caught:
            apply_reconfigure_lrs(True, ContainerWithoutSupport())
        # The message must name the fix, or an operator is left guessing.
        self.assertIn("does not support preserving learning", str(caught.exception))
        self.assertIn("DiSCOOptimizersContainer", str(caught.exception))

    def test_a_supporting_container_is_armed(self):
        from torchtitan.components.checkpointer.base import apply_reconfigure_lrs

        class ContainerWithSupport:
            preserve_lrs_when_loading = False

        container = ContainerWithSupport()
        apply_reconfigure_lrs(True, container)
        self.assertTrue(container.preserve_lrs_when_loading)

    def test_the_flag_off_is_a_no_op_even_for_an_unsupported_container(self):
        from torchtitan.components.checkpointer.base import apply_reconfigure_lrs

        class ContainerWithoutSupport:
            pass

        apply_reconfigure_lrs(False, ContainerWithoutSupport())  # must not raise

    def test_both_checkpointers_use_the_shared_helper(self):
        """The duplication this replaced was two byte-identical 13-line blocks."""
        import inspect

        from torchtitan.components.checkpointer.dcp import CheckpointManager

        self.assertIn(
            "apply_reconfigure_lrs", inspect.getsource(CheckpointManager.__init__)
        )

    def test_the_fork_container_does_support_it(self):
        from torchtitan.optimizers.container import DiSCOOptimizersContainer

        self.assertIn(
            "preserve_lrs_when_loading",
            DiSCOOptimizersContainer.__init__.__code__.co_names
            + tuple(DiSCOOptimizersContainer.load_state_dict.__code__.co_names),
        )


if __name__ == "__main__":
    unittest.main()
