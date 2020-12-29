# File containing utilities
from typing import Callable

import torch

from hyperparameters import SETTINGS


def reportedRun(title: str, fun: Callable[[], None]):
    print(f"{title}...")
    result = fun()
    print(f"{title} Done.")
    return result


def debug(*args, **kwargs):
    if SETTINGS.DEBUG:
        print(*args, **kwargs)


def map_1D(func, lst):
    return list(map(func, lst))


def map_2D(func, matrix):
    return list(map(lambda row: map_1D(func, row), matrix))


def withinError(tensor1, tensor2, relativeError=0.01):
    # Error relative to tensor 1
    diff = tensor1 - tensor2
    bools = abs(diff) < abs(tensor1 * relativeError)
    if len(bools) == bools.sum():
        return True
    return False


def maxError(tensor1, tensor2):
    # Error relative to tensor 1
    relError = abs((tensor1 - tensor2) / tensor1)
    inverseMask = relError.eq(relError)
    # TODO - if there are no non-nan values this will error
    return relError[inverseMask].max()


def nantensor(size):
    zeros = torch.zeros(size)
    return zeros / zeros


'''
This will take the masked min
In the event that there is no min in a dimension the identity will be the
absolute min of the tensor
'''


def maskedmax(tensor, mask, dim=0):
    mask = mask.logical_not()  # True when needs to be masked out
    absmin = tensor.min()  # absolute min
    absmax = tensor.max()  # absolute max
    mask = mask.type(torch.Tensor) * (absmin - absmax)  # (absmin-abssmax) when needs to be masked out, 0 when kept
    tensor += mask  # doesn't affect non-masked out, maskedout values have min added, max removed. They will be made at least as low as the lowest value
    return tensor.max(dim)
