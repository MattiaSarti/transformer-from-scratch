"""
Utilities for selecting devices.
"""


from torch.cuda import is_available as cuda_is_available


def select_device(gpu_if_possible: bool):
    """
    Select the device handling computations based on whether a GPU is
    requested and whether it is available.
    """
    # employing a GPU if requested and possible:
    if gpu_if_possible and cuda_is_available():
        return 'cuda:0'
    return 'cpu'
