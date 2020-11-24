# Data structure for containing hyper-parameters, as well as pre-defined hyper-parameter 'bundles' for ease-of-use

"""Used to store hyperparameters/settings"""
from enum import Enum, auto

import datasets
from datastructures import Dataset


class NormalisationMethod(Enum):
    RelNorm = auto()
    MentNorm = auto()

class Hyperparameters:  # TODO - create hyper-parameter settings object
    dataDir = "../data/"
    dataDir_csv = "../data/generated/test_train_data/"
    training: bool = True
    dataset: Dataset = datasets.train_AIDA
    normalisation = NormalisationMethod.MentNorm
    d: int = 300
    gamma: float = 0.01
    LBP_loops: int = 10  # TODO is this needed?
    dropout_rate: float = 0.3
    window_size: int = 6
    k: int = 3
    rel_specialinit: bool = True
    learning_reduction_threshold_f1: float = 0.915
    learning_rate_initial: float = 10E-4
    learning_rate_final: float = 10E-5
    learning_stop_threshold_epochs: int = 20

SETTINGS = Hyperparameters()
BUNDLE_relNorm = Hyperparameters()
BUNDLE_mentNorm = Hyperparameters()
BUNDLE_mentNormK1 = Hyperparameters()
BUNDLE_mentNormNoPad = Hyperparameters()

BUNDLE_relNorm.normalisation = NormalisationMethod.RelNorm
BUNDLE_relNorm.rel_specialinit = False
BUNDLE_relNorm.k = 6
BUNDLE_relNorm.learning_reduction_threshold_f1 = 0.91

BUNDLE_mentNormK1.k = 1

#TODO mentNormNoPad needs some parameters set

# TODO - specify useful hyper-parameter bundles
