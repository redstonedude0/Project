# This file obtains the training and evaluation datasets and prepares it for use.
#
# Note: Data provided has been pre-computed with p_e_m probabilities
# Data is from sources:
# training set:
#   AIDA-CoNLL>AIDA-train - 946 docs
# dev evaluation set:
#   AIDA-CoNLL>AIDA-A - 216 docs
# test evaluation sets:
#   AIDA-CoNLL>AIDA-B - 231 docs
#   MSNBC - 20 docs
#   AQUAINT - 50 docs
#   ACE2004 - 36 docs
#   WNED-CWEB(CWEB) - 320 docs
#   WNED-WIKI(WIKI) - 320 docs
#
# AIDA sets are considered in-domain, other sets are considered out-domain.
#

from enum import Enum, auto


class Dataset(Enum):
    # TODO - replace auto() call with a function which generates a Dataset object instead, which loads in each dataset
    train_AIDA = auto()
    dev_AIDA = auto()
    test_AIDA = auto()
    test_MSNBC = auto()
    test_AQUAINT = auto()
    test_ACE2004 = auto()
    test_CWEB = auto()
    test_WIKI = auto()
