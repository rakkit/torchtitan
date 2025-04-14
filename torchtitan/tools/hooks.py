# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def print_in_backward_hook(name):
    """
    register a hook to print the gradient of the tensor in backward.

    For parameter tensor of model, you can do:
    ```
        for name, param in model_parts[0].named_parameters():
            if param.requires_grad:
                param.register_hook(print_grad_hook(name))
    ```

    and any (intermediate) tensor that you want to print the gradient, you can do:
    ```
        tensor.retain_grad()
        tensor.register_hook(print_in_backward_hook(name))
    ```
    """

    def hook(grad):
        if grad is None:
            print(f"[Backward Hook] Backward into {name}: No gradient!")
        else:
            print(
                f"[Backward Hook] Backward into {name}: grad shape={grad.shape}, mean={grad.mean().item():.6f}"
            )
        return grad

    return hook
