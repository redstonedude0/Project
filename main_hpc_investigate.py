# Main pipeline to run the paper, but for HPC usage

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
from hyperparameters import SETTINGS

SETTINGS.data_dir = "/rds/user/hrjh2/hpc-work/"
SETTINGS.data_dir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
SETTINGS.data_dir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddingss/word_ent_embs/"
SETTINGS.data_dir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
SETTINGS.low_mem = False

import files
import processeddata
from utils import *

reported_run("Checking Datadir", files.check_data_dir)
reported_run("Loading embeddingss", processeddata.load_embeddings)

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)
import torch.nn as nn

gpu = torch.device("cuda:0")
x = nn.Sequential(
    nn.Linear(900, 300).float(),  # TODO what dimensions?
    nn.Tanh(),
    nn.Dropout(p=SETTINGS.dropout_rate),
).float().to(gpu)
test = torch.randn(900).to(gpu)
test.requires_grad = True
y = x(test)
z = y.sum()
z.backward()
print("Test complete.")
