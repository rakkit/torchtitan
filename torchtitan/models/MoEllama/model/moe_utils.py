# https://arxiv.org/abs/2502.16982
import numpy as np

# SCALING_FACTORS = {
#     (4, 1): 1.0,
#     (4, 2): 1.3975240382831564,
#     (4, 4): 1.8746839497244312,
#     (8, 1): 1.0,
#     (8, 2): 1.4089312089643131,
#     (8, 4): 1.9716372098310582,
#     (8, 6): 2.3692189889782083,
#     (8, 8): 2.6297724148600703,
#     (16, 1): 1.0,
#     (16, 2): 1.412131787337075,
#     (16, 4): 1.990941161710805,
#     (16, 6): 2.4283171262135257,
#     (16, 8): 2.7879495517358537,
#     (16, 16): 3.7047009687546923,
#     (32, 1): 1.0,
#     (32, 2): 1.4132875875185074,
#     (32, 4): 1.9964881494834583,
#     (32, 6): 2.4419467013356364,
#     (32, 8): 2.8154055554895394,
#     (32, 16): 3.943324881680555,
#     (32, 32): 5.229939776738365,
#     (64, 1): 1.0,
#     (64, 2): 1.4137804787923913,
#     (64, 4): 1.9984419912715523,
#     (64, 6): 2.446338919929975,
#     (64, 8): 2.8232884471713273,
#     (64, 16): 3.981865423254339,
#     (64, 32): 5.5779444133949685,
#     (64, 64): 7.391360398907228,
#     (128, 1): 1.0,
#     (128, 2): 1.4139935847474265,
#     (128, 4): 1.999243359037702,
#     (128, 6): 2.448026924377305,
#     (128, 8): 2.826111482320644,
#     (128, 16): 3.992795051245798,
#     (128, 32): 5.631330086465613,
#     (128, 64): 7.8888558415100025,
#     (128, 128): 10.447601374558701,
#     (256, 1): 1.0,
#     (256, 2): 1.4140933478999065,
#     (256, 4): 1.9996002066706609,
#     (256, 6): 2.4487450517795386,
#     (256, 8): 2.827297937808976,
#     (256, 16): 3.996711691215216,
#     (256, 32): 5.646620595072715,
#     (256, 64): 7.964034924039261,
#     (256, 128): 11.157350249127516,
# }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_gate_scaling_factor(
    num_experts: int, topk: int, iter_times: int = 10000
) -> float:
    if num_experts < 1 or topk < 1:
        return 1.0
    """
    Calculate the gate scaling factor for Mixture-of-Experts (MoE).

    Args:
        num_experts (int): Total number of experts.
        topk (int): Number of top experts selected by the router.
        iter_times (int): Number of iterations to estimate the expected scaling factor.

    Returns:
        float: Estimated gate scaling factor.
    """
    factors = []

    for _ in range(iter_times):
        # Simulate Gaussian-distributed logits
        logits = np.random.randn(num_experts)

        # Apply sigmoid to get scores in [0, 1]

        p = np.sort(sigmoid(logits))[::-1]
        p = p[:topk]
        p = p / p.sum()
        scaling_factor = 1.0 / (p**2).sum() ** 0.5
        factors.append(scaling_factor)

    return np.mean(factors)


# Example test
if __name__ == "__main__":
    num_experts = 16
    topk = 2
    iter_times = 10000

    scale = calc_gate_scaling_factor(num_experts, topk, iter_times)

    for num_experts in [4, 8, 16, 32, 64, 128, 256]:
        for topk in [1, 2, 4, 6, 8, 16, 32, 64, 128, 256]:
            if topk > num_experts:
                continue
            scale = calc_gate_scaling_factor(num_experts, topk, iter_times)
            print(f"num_experts: {num_experts}, topk: {topk}, scale: {scale}")
