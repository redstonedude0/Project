# File containing utilities
import sys
from typing import Callable

import torch

from hyperparameters import SETTINGS


def reported_run(title: str, fun: Callable[[], None]):
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


def max_error(tensor1, tensor2):
    # Error relative to tensor 1
    rel_error = abs((tensor1 - tensor2) / tensor1)
    inverse_mask = rel_error.eq(rel_error)
    # TODO - if there are no non-nan values this will error
    return rel_error[inverse_mask].max()


def sum_error(tensor1, tensor2):
    # Error relative to tensor 1
    rel_error = abs((tensor1 - tensor2))
    inverse_mask = rel_error.eq(rel_error)
    # TODO - if there are no non-nan values this will error
    return rel_error[inverse_mask].sum()


def max_error_masked(tensor1, tensor2, mask):
    # Error relative to tensor 1
    rel_error = abs((tensor1 - tensor2) / tensor1)
    inverse_mask = rel_error.eq(rel_error)
    inverse_mask = inverse_mask.logical_and(mask)  # True iff keep
    # TODO - if there are no non-nan values this will error
    return rel_error[inverse_mask].max()

'''
This will take the masked min
In the event that there is no min in a dimension the identity will be the
absolute min of the tensor
'''

'''
max over a tensor, nan values are ignored (unless entire thing nans, then result is nan)
'''


def smart_max(tensor, dim=-1):
    if dim == -1:
        return tensor[tensor == tensor].max()
    ninf = float("-inf")
    nan = float("nan")
    tensor = tensor.clone()  # dont change original tensor
    tensor[tensor != tensor] = ninf  # make nans negative inf for purposes of finding max
    max_tensor = tensor.max(dim)[0]  # values only
    max_tensor[max_tensor == ninf] = nan  # convert ninfs to nans
    return max_tensor


'''
Sum over a tensor, nan values are ignored (treated as 0s, unless entire thing is nan then result is nan)
'''


def smart_sum(tensor, dim=-1):
    if dim == -1:
        # TODO: ERROR! If dim=-1 and entire tensor is nans then sum is not nan as specced (unity error)
        return tensor[tensor == tensor].sum()
    tensor_ = tensor.clone()  # dont change original tensor
    tensor_[tensor_ != tensor_] = 0  # make nans 0 for purposes of finding sum
    sum_tensor = tensor_.sum(dim)
    # Replace sums with nans if the summing dim was only nans:
    tensor_ = torch.zeros(tensor.shape)  # create same shape tensor of ones
    nan = float("nan")
    tensor_[tensor == tensor] = nan  # where tensor is number, make tensor_ nan
    tensor_ = tensor_.sum(dim)  # nan if row contained atleast 1 numeral, 0 otherwise
    sum_tensor[tensor_ == tensor_] = nan  # nan where tensor row contained no numerals
    return sum_tensor

'''
normalise in-place a tensor (with a masks which should be False for all nan/exclude values) such that the average of non-masked values is translated to zero
mask should be shape broadcastable across tensor
final tensor has non-masked values normalised to have average 0
masked values in final tensor are all 0
normalisation is row-wise with the specific dim the one normalised across
'''


# TODO - is mask polarity inverted? (do other functions use opposite mask polarity?)

def normalise_avg_to_zero_rows(tensor, broadcastable_mask, dim=0):
    mask = broadcast_mask(tensor, broadcastable_mask)
    tensor[~mask] = 0  # set tensor to 0 where masked out
    tensor_copy = tensor.clone()
    tensor_copy[mask] = 1  # set copy to 1 where not masked out
    # same dim(broadcastable) count of the non masked and total per row
    non_masked_count = tensor_copy.sum(dim=dim, keepdims=True)
    numerical_total = tensor.sum(dim=dim, keepdims=True)
    # same dim(broadcastable) avg across the row (if nMC is 0 then avg is +- inf, will be reset to 0)
    avg = numerical_total / non_masked_count
    tensor -= avg  # subtract average to make new average 0 (broadcasts)
    tensor[~mask] = 0  # reset masked values to 0
    # done


'''
IN-PLACE
Sets the value of tensor to the masked_value when the broadcastable mask is true
(when broadcasted to make shapes match)
TODO - we assume similar shape (same dims) does broadcasting properly assume this? (terminology check)
'''


def set_mask_broadcastable(tensor, broadcastable_mask, masked_value):
    mask = broadcast_mask(tensor, broadcastable_mask)
    tensor[mask] = masked_value  # set tensor to value when mask


'''
Calculte the broadcasted mask and return it
TODO - terminology (as above)
'''


def broadcast_mask(tensor, broadcastable_mask):
    # torch.Size -> torch.tensor
    mask_size = torch.tensor(broadcastable_mask.shape)
    tensor_size = torch.tensor(tensor.shape)
    # integer division, should be whole so // is fine
    # torch.tensor->python list
    repeat_size = (tensor_size // mask_size).tolist()
    return broadcastable_mask.repeat(repeat_size)

def nan_test(tensor, title):
    if len(tensor[tensor != tensor]) >= 1:
        msg = f"Nans detected at '{title}'"
        print(msg)
        print(msg,file=sys.stderr)
        print("Tensor:",tensor)

#Stopwords from Le et al.
STOP_WORDS = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all',
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
def is_important_dict_word(s,special=None):
    try:
        if len(s) <= 1 or s.lower() in STOP_WORDS:
            return False
        float(s)
        return False
    except:
        #Is important word, is it actually in dict?
        import processeddata
        if special == "snd":
            if s in processeddata.word_to_word_id_snd:
                return True
        else:
            if s in processeddata.word_to_word_id:
                return True
        return False


'''
Normalise a token, as the Vocabulary object from Le et al. does.
'''
BRACKET_MAP = {"-LCB-": "{", "-LRB-": "(", "-LSB-": "[", "-RCB-": "}", "-RRB-": ")", "-RSB-": "]"}
def token_normalise(t):
    if t in ["#UNK#", "<s>", "</s>"]:
        return t#If unknown just return
    if t in BRACKET_MAP.keys():
        return BRACKET_MAP[t]#Convert brackets
    return t

'''
Convet a string to the embeddingss of the surrounding context, the same way the implementation by Le et al. does.
'''
def string_to_token_embeddings(s, trim="none", window_size = -1, special=None):
    import processeddata
    tokens = s.strip().split()
    tokens = [token_normalise(t) for t in tokens]
    if special == "snd":
        embeddings = [processeddata.word_id_to_embedding_snd[processeddata.word_to_word_id_snd[token]] for token in tokens if
                      is_important_dict_word(token,special="snd")]
    else:
        embeddings = [processeddata.word_id_to_embedding[processeddata.word_to_word_id[token]] for token in tokens if
                      is_important_dict_word(token)]
    if trim == "left":
        embeddings = embeddings[-(window_size // 2):]
    elif trim == "right":
        embeddings = embeddings[:(window_size // 2)]
    elif trim != "none":
        raise Exception(f"Unknown trim value {trim}")
    return embeddings

def string_to_token_ids(s, trim="none", window_size = -1, special=None):
    import processeddata
    tokens = s.strip().split()
    tokens = [token_normalise(t) for t in tokens]
    if special == "snd":
        ids = [processeddata.word_to_word_id_snd[token] for token in tokens if
               is_important_dict_word(token,special="snd")]
    else:
        ids = [processeddata.word_to_word_id[token] for token in tokens if
               is_important_dict_word(token)]
    if trim == "left":
        ids = ids[-(window_size // 2):]
    elif trim == "right":
        ids = ids[:(window_size // 2)]
    elif trim != "none":
        raise Exception(f"Unknown trim value {trim}")
    return ids