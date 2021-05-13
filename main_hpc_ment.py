# Main pipeline to run the paper, but for HPC usage

import datastructures

datastructures.SETTINGS  # reference to prevent optimise away
from hyperparameters import SETTINGS

SETTINGS.data_dir = "/rds/user/hrjh2/hpc-work/"
SETTINGS.data_dir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
SETTINGS.data_dir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddingss/word_ent_embs/"
SETTINGS.data_dir_checkpoints = "/rds/user/hrjh2/hpc-work/checkpoints/"
SETTINGS.low_mem = False
SETTINGS.save_name = "save_ment_OLD"
quit(2)
import main

print("HPC Results:")
main.model.evals.print()
