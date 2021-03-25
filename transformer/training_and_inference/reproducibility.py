"""
Utilities for ensuring reproducible results.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available as cuda_is_available
from torch.backends import cudnn


def make_results_reproducible(model_is_convolutional: bool = False) -> None:
    """
    Making the subsequent instructions produce purely deterministic outputs
    by fixing all the relevant seeds:
    """
    random_seed(0)
    _ = numpy_seed(0)
    _ = torch_manual_seed(0)

    # since GPU computations may introduce additional stochasticity with
    # their convolutional operation optimizations:
    if model_is_convolutional:
        if cuda_is_available():
            # disabling benchmarking and choosing among convolution operation
            # implementation alternatives, which is stochastic based due to
            # noise and hardware:
            cudnn.benchmark = False
            # ansuring deterministic algorithms for colvolutional operations
            # are employed:
            cudnn.deterministic = True
