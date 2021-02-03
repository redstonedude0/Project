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
    SETTINGS.dataset = datasets.loadDataset("aida_train.csv")
    print(f"Size of training dataset: {len(SETTINGS.dataset.documents)}")
    # For debug SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:10]
    model = modeller.trainToCompletion()
else:
    pass  # TODO - eval
