# Main pipeline to run the paper, but for HPC usage

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
from hyperparameters import SETTINGS

SETTINGS.dataDir = "/rds/user/hrjh2/hpc-work/"
SETTINGS.dataDir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
SETTINGS.dataDir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddings/word_ent_embs/"
SETTINGS.dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
SETTINGS.lowmem = False

import files
import processeddata
from utils import *

reportedRun("Checking Datadir", files.checkDataDir)
reportedRun("Loading embeddings", processeddata.loadEmbeddings)

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)
import torch.nn as nn

x = nn.Sequential(
    nn.Linear(900, 300).float(),  # TODO what dimensions?
    nn.Tanh(),
    nn.Dropout(p=SETTINGS.dropout_rate),
).float().to(torch.device("gpu:0"))
test = torch.randn(900)
test.requires_grad = True
y = x(test)
y.backward()
print("Test complete.")
