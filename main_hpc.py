# Main pipeline to run the paper, but for HPC usage

from hyperparameters import SETTINGS

SETTINGS.dataDir = "/rds/user/hrjh2/hpc-work/"
SETTINGS.dataDir_csv = "/rds/user/hrjh2/hpc-work/generated/test_train_data/"
SETTINGS.dataDir_embeddings = "/rds/user/hrjh2/hpc-work/generated/embeddings/word_ent_embs/"
import main

print(main.results)
