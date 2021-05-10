# Data structure for containing hyper-parameters, as well as pre-defined hyper-parameter 'bundles' for ease-of-use

"""Used to store hyperparameters/settings"""
from enum import Enum, auto


class NormalisationMethod(Enum):
    RelNorm = auto()
    MentNorm = auto()


class Hyperparameters:
    dataDir = "/home/harrison/Documents/project/data/"
    dataDir_csv = "/home/harrison/Documents/project/data/generated/test_train_data/"
    dataDir_conll = "/home/harrison/Documents/project/data/basic_data/test_datasets/"
    dataDir_personNames = "/home/harrison/Documents/project/data/basic_data/p_e_m_data/persons.txt"
    dataDir_embeddings = "/home/harrison/Documents/project/data/generated/embeddings/word_ent_embs/"
    dataDir_checkpoints = "/home/harrison/Documents/project/data/checkpoints/"
    training: bool = True
    dataset_train: 'Dataset' = None
    dataset_eval: 'Dataset' = None
    normalisation = NormalisationMethod.RelNorm  # TODO change back to mentnorm once it's implemented
    d: int = 300
    gamma: float = 0.01
    LBP_loops: int = 10
    dropout_rate: float = 0.3
    k: int = 3
    rel_specialinit: bool = True
    learning_reduction_threshold_f1: float = 0.91
    learning_rate_initial: float = 1E-4
    learning_rate_final: float = 1E-5
    learning_stop_threshold_epochs: int = 20
    DEBUG: bool = True
    lambda1: float = -1E-7
    lambda2: float = -1E-7
    allow_nans = False  # True if nans should be used to represent undefined values, false if not
    lowmem = True  # True if need to use lowmem settings
    device = None  # Used to store the device for training
    saveName = "save_default"
    loss_patched = False #Use loss&accuracy functions from the paper for certainty of comparison
    attention_token_count = 25
    context_window_size = 100#main window
    context_window_prerank = 50
    context_window_fmc = 6
    pad_candidates = True
    n_cands_pem = 4
    n_cands_ctx = 4
    n_cands = 8 #Should be n_cands_pem+n_cands_ctx
    switches = {
        "aug_conll":True,
        "aug_coref":True,
        "switch_sel":True,
        "override_embs":True,
        "consistency_psi":True,
        "pad_enable":False,
        "snd_embs":True,
        "exp_adjust":True
    }
    SEED: int = 0
    def __repr__(self):
        s = "Hyperparameters:"
        keys = dir(self)
        for key in keys:
            if not key.startswith("__"):
                #Not internal
                s += f"\n  {key}:{getattr(self,key)}"
        return s


SETTINGS = Hyperparameters()

def APPLYBUNDLE_relNorm(settings:Hyperparameters):
    settings.normalisation = NormalisationMethod.RelNorm
    settings.rel_specialinit = True
    settings.k = 6
    settings.learning_reduction_threshold_f1 = 0.91

def APPLYBUNDLE_mentNorm(settings:Hyperparameters):
    settings.normalisation = NormalisationMethod.MentNorm
    settings.rel_specialinit = False
    settings.k = 3
    settings.learning_reduction_threshold_f1 = 0.915

def APPLYBUNDLE_mentNormK1(settings:Hyperparameters):
    APPLYBUNDLE_mentNorm(settings)
    settings.k = 1

def APPLYBUNDLE_mentNormNoPad(settings:Hyperparameters):
    # TODO mentNormNoPad needs some parameters set
    APPLYBUNDLE_mentNorm(settings)
    pass

def APPLYBUNDLE_hpc(settings:Hyperparameters):
    settings.dataDir = "/rds/user/hrjh2/hpc-work/"
    settings.dataDir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
    settings.dataDir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddings/word_ent_embs/"
    settings.dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
    settings.dataDir_conll = "/rds/user/hrjh2/hpc-work/basic_data/test_datasets/"
    settings.dataDir_personNames = "/rds/user/hrjh2/hpc-work/basic_data/p_e_m_data/persons.txt"
    settings.lowmem = True

def APPLYBUNDLE_colab(settings:Hyperparameters):
    settings.dataDir = "drive/MyDrive/project/data/"
    settings.dataDir_csv = "drive/MyDrive/project/data/generated/test_train_data/"
    settings.dataDir_embeddings = "drive/MyDrive/project/data/generated/embeddings/word_ent_embs/"
    settings.dataDir_checkpoints = "drive/MyDrive/project/data/checkpoints/"
    settings.dataDir_conll = "drive/MyDrive/project/data/basic_data/test_datasets/"
    settings.dataDir_personNames = "drive/MyDrive/project/data/basic_data/p_e_m_data/persons.txt"
    settings.lowmem = False

def APPLYBUNDLE_paper(settings:Hyperparameters):
    settings.loss_patched = True

def APPLYBUNDLE_blind(settings:Hyperparameters):
    settings.switches = {#Disable all switches
        "aug_conll":False,
        "aug_coref":False,
        "switch_sel":False,
        "override_embs":False,
        "consistency_psi":False,
        "pad_enable":False,
        "snd_embs":False,
        "exp_adjust":False
    }

def APPLYBUNDLE_blindN(settings:Hyperparameters,n=0):
    settings.switches = {#Disable all switches
        "aug_conll":n==0,
        "aug_coref":n==1,
        "switch_sel":n==2,
        "override_embs":n==3,
        "consistency_psi":n==4,
        "pad_enable":n==5,
        "snd_embs":n==0,#moved to switch 0 instead
        "exp_adjust":n==7
    }


# TODO - specify useful hyper-parameter bundles
