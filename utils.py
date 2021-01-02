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


def maxErrorMasked(tensor1, tensor2, mask):
    # Error relative to tensor 1
    relError = abs((tensor1 - tensor2) / tensor1)
    inverseMask = relError.eq(relError)
    inverseMask = inverseMask.logical_and(mask)  # True iff keep
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


'''
max over a tensor with a mask, nan values are ignored
'''


def maskedmax(tensor, mask, dim=-1):
    mask = mask.logical_not()  # True when needs to be masked out
    absmin = tensor[tensor == tensor].min()  # absolute min (ignoring nans)
    absmax = tensor[tensor == tensor].max()  # absolute max
    mask = mask.type(torch.Tensor) * (absmin - absmax)  # (absmin-abssmax) when needs to be masked out, 0 when kept
    tensor += mask  # doesn't affect non-masked out, maskedout values have min added, max removed. They will be made at least as low as the lowest value
    if dim == -1:
        return tensor[tensor == tensor].max()
    ninf = float("-inf")
    nan = float("nan")
    tensor = tensor.clone()  # dont change original tensor
    tensor[tensor != tensor] = ninf  # make nans negative inf for purposes of finding max
    maxTensor = tensor.max(dim)[0]  # values only
    maxTensor[maxTensor == ninf] = nan  # convert ninfs to nans
    return maxTensor


'''
Sum over a tensor with a mask, nan values are ignored
'''


def maskedsum(tensor, mask, dim=-1):
    mask = mask.type(torch.Tensor)  # 0 when needs to be masked out, 1 when kept
    tensor *= mask  # doesn't affect non-masked out, maskedout values now equal 0
    if dim == -1:
        return tensor[tensor == tensor].sum()
    tensor = tensor.clone()  # dont change original tensor
    tensor[tensor != tensor] = 0  # make nans 0 for purposes of finding sum
    sumTensor = tensor.sum(dim)
    return sumTensor  # If summing nans a default of 0 will have to do


'''
max over a tensor, nan values are ignored (unless entire thing nans, then result is nan)
'''


def smartmax(tensor, dim=-1):
    if dim == -1:
        return tensor[tensor == tensor].max()
    ninf = float("-inf")
    nan = float("nan")
    tensor = tensor.clone()  # dont change original tensor
    tensor[tensor != tensor] = ninf  # make nans negative inf for purposes of finding max
    maxTensor = tensor.max(dim)[0]  # values only
    maxTensor[maxTensor == ninf] = nan  # convert ninfs to nans
    return maxTensor


'''
Sum over a tensor, nan values are ignored (treated as 0s, unless entire thing is nan then result is nan)
'''


def smartsum(tensor, dim=-1):
    if dim == -1:
        # TODO: ERROR! If dim=-1 and entire tensor is nans then sum is not nan as specced (unity error)
        return tensor[tensor == tensor].sum()
    tensor_ = tensor.clone()  # dont change original tensor
    tensor_[tensor_ != tensor_] = 0  # make nans 0 for purposes of finding sum
    sumTensor = tensor_.sum(dim)
    # Replace sums with nans if the summing dim was only nans:
    tensor_ = torch.zeros(tensor.shape)  # create same shape tensor of ones
    nan = float("nan")
    tensor_[tensor == tensor] = nan  # where tensor is number, make tensor_ nan
    tensor_ = tensor_.sum(dim)  # nan if row contained atleast 1 numeral, 0 otherwise
    sumTensor[tensor_ == tensor_] = nan  # nan where tensor row contained no numerals
    return sumTensor  # If summing nans a default of 0 will have to do
