# Main pipeline to run the paper, but for HPC usage

import datastructures

datastructures.SETTINGS  # reference to prevent optimise
from hyperparameters import SETTINGS

SETTINGS.dataDir = "/rds/user/hrjh2/hpc-work/"
SETTINGS.dataDir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
SETTINGS.dataDir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddings/word_ent_embs/"
SETTINGS.dataDir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
SETTINGS.lowmem = False
import main

print("HPC Results:")
main.results.print()
