# Main pipeline to run the paper

import datasets
import files
import modeller
import processeddata
import torch
from utils import *

print("Cuda?", torch.cuda.is_available())
reportedRun("Checking Datadir", files.checkDataDir)
reportedRun("Loading embeddings", processeddata.loadEmbeddings)

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)

if SETTINGS.training:
    SETTINGS.dataset_train = datasets.loadDataset("aida_train.csv","AIDA/aida_train.txt")
    SETTINGS.dataset_eval = datasets.loadDataset("aida_testA.csv","AIDA/testa_testb_aggregate_original")
    print(f"Size of training dataset: {len(SETTINGS.dataset_train.documents)}")
    print(f"Size of eval dataset: {len(SETTINGS.dataset_eval.documents)}")
    # For debug SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:10]
    model = modeller.trainToCompletion()
else:
    pass  # TODO - eval
