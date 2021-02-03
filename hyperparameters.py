# Data structure for containing hyper-parameters, as well as pre-defined hyper-parameter 'bundles' for ease-of-use

"""Used to store hyperparameters/settings"""
from enum import Enum, auto


class NormalisationMethod(Enum):
    RelNorm = auto()
    MentNorm = auto()


class Hyperparameters:
    dataDir = "/home/harrison/Documents/project/data/"
    dataDir_csv = "/home/harrison/Documents/project/data/generated/test_train_data/"
    dataDir_embeddings = "/home/harrison/Documents/project/data/generated/embeddings/word_ent_embs/"
    dataDir_checkpoints = "/home/harrison/Documents/project/data/checkpoints/"
    training: bool = True
    dataset: 'Dataset' = None
    normalisation = NormalisationMethod.RelNorm  # TODO change back to mentnorm once it's implemented
    d: int = 300
    gamma: float = 0.01
    LBP_loops: int = 10
    dropout_rate: float = 0.3
    window_size: int = 6
    k: int = 3
    rel_specialinit: bool = True
    learning_reduction_threshold_f1: float = 0.91
    learning_rate_initial: float = 10E-4
    learning_rate_final: float = 10E-5
    learning_stop_threshold_epochs: int = 20
    DEBUG: bool = True
    lambda1: float = -1E-7
    lambda2: float = -1E-7
    allow_nans = False  # True if nans should be used to represent undefined values, false if not
    lowmem = True  # True if need to use lowmem settings
    device = None  # Used to store the device for training
    saveName = "save_default"
    loss_patched = False #Use loss function from the paper for certainty of comparison


SETTINGS = Hyperparameters()

def APPLYBUNDLE_relNorm(settings:Hyperparameters):
    settings.normalisation = NormalisationMethod.RelNorm
    settings.rel_specialinit = True
    settings.k = 6
    settings.learning_reduction_threshold_f1 = 0.91

def APPLYBUNDLE_mentNorm(settings:Hyperparameters):
    settings.normalisation = NormalisationMethod.MentNorm
    settings.rel_specialinit = False
    settings.learning_reduction_threshold_f1 = 0.915

def APPLYBUNDLE_mentNormK1(settings:Hyperparameters):
    APPLYBUNDLE_mentNorm(settings)
    settings.k = 1

def APPLYBUNDLE_mentNormNoPad(settings:Hyperparameters):
    # TODO mentNormNoPad needs some parameters set
    pass

def APPLYBUNDLE_hpc(settings:Hyperparameters):
    settings.dataDir = "/rds/user/hrjh2/hpc-work/"
    settings.dataDir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
    settings.dataDir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddings/word_ent_embs/"
    settings.dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
    settings.lowmem = False
    pass

# TODO - specify useful hyper-parameter bundles
