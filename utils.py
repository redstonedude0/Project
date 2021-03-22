# File containing utilities
import sys
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


def sumError(tensor1, tensor2):
    # Error relative to tensor 1
    relError = abs((tensor1 - tensor2))
    inverseMask = relError.eq(relError)
    # TODO - if there are no non-nan values this will error
    return relError[inverseMask].sum()


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
    return sumTensor


'''
normalise in-place a tensor (with a masks which should be False for all nan/exclude values) such that the average of non-masked values is translated to zero
mask should be shape broadcastable across tensor
final tensor has non-masked values normalised to have average 0
masked values in final tensor are all 0
'''


# TODO - deprecated
def normalise_avgToZero(tensor, broadcastable_mask):
    mask = broadcastMask(tensor, broadcastable_mask)
    tensor[~mask] = 0  # set tensor to 0 where masked out
    tensor_copy = tensor.clone()
    tensor_copy[mask] = 1  # set copy to 1 where not masked out
    nonMaskedCount = tensor_copy.sum()
    numericalTotal = tensor.sum()
    avg = numericalTotal / nonMaskedCount
    tensor -= avg  # subtract average to make new average 0
    tensor[~mask] = 0  # reset masked values to 0
    # done


'''
normalise in-place a tensor (with a masks which should be False for all nan/exclude values) such that the average of non-masked values is translated to zero
mask should be shape broadcastable across tensor
final tensor has non-masked values normalised to have average 0
masked values in final tensor are all 0
normalisation is row-wise with the specific dim the one normalised across
'''


# TODO - is mask polarity inverted? (do other functions use opposite mask polarity?)

def normalise_avgToZero_rowWise(tensor, broadcastable_mask, dim=0):
    mask = broadcastMask(tensor, broadcastable_mask)
    tensor[~mask] = 0  # set tensor to 0 where masked out
    tensor_copy = tensor.clone()
    tensor_copy[mask] = 1  # set copy to 1 where not masked out
    # same dim(broadcastable) count of the non masked and total per row
    nonMaskedCount = tensor_copy.sum(dim=dim, keepdims=True)
    numericalTotal = tensor.sum(dim=dim, keepdims=True)
    # same dim(broadcastable) avg across the row (if nMC is 0 then avg is +- inf, will be reset to 0)
    avg = numericalTotal / nonMaskedCount
    tensor -= avg  # subtract average to make new average 0 (broadcasts)
    tensor[~mask] = 0  # reset masked values to 0
    # done


'''
IN-PLACE
Sets the value of tensor to the maskedValue when the broadcastable mask is true
(when broadcasted to make shapes match)
TODO - we assume similar shape (same dims) does broadcasting properly assume this? (terminology check)
'''


def setMaskBroadcastable(tensor, broadcastable_mask, maskedValue):
    mask = broadcastMask(tensor, broadcastable_mask)
    tensor[mask] = maskedValue  # set tensor to value when mask


'''
Calculte the broadcasted mask and return it
TODO - terminology (as above)
'''


def broadcastMask(tensor, broadcastable_mask):
    # torch.Size -> torch.tensor
    masksize = torch.tensor(broadcastable_mask.shape)
    tensorsize = torch.tensor(tensor.shape)
    # integer division, should be whole so // is fine
    # torch.tensor->python list
    repeatsize = (tensorsize // masksize).tolist()
    return broadcastable_mask.repeat(repeatsize)

def nantest(tensor,title):
    if len(tensor[tensor != tensor]) >= 1:
        msg = f"Nans detected at '{title}'"
        print(msg)
        print(msg,file=sys.stderr)
        print("Tensor:",tensor)

#Stopwords from Le et al.
STOPWORDS = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all',
             'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
             'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
             'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be',
             'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
             'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom',
             'but', 'by', 'call', 'can', 'cannot', 'cant', 'dont', 'co', 'con', 'could', 'couldnt',
             'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
             'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
             'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty',
             'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
             'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred',
             'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself',
             'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may',
             'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless',
             'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
             'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
             'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per',
             'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six',
             'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
             'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
             'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
             'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though',
             'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very',
             'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
             'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
             'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
             'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves',
             'st', 'years', 'yourselves', 'new', 'used', 'known', 'year', 'later', 'including', 'used',
             'end', 'did', 'just', 'best', 'using']

'''
Checks if a word is considered 'important' by Le et al.
'''
def is_important_dict_word(s):
    try:
        if len(s) <= 1 or s.lower() in STOPWORDS:
            return False
        float(s)
        return False
    except:
        #Is important word, is it actually in dict?
        import processeddata
        if s in processeddata.word2wordid:
            return True
        return False