# Main pipeline to run the paper

import datasets
import files
import modeller
from hyperparameters import SETTINGS
from utils import *

reportedRun("Checking Datadir", lambda: files.checkDataDir())  # TODO - fill in dir
reportedRun("Loading Datasets", lambda: datasets.loadDatasets())

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)

if SETTINGS.training:
    model, results = modeller.trainToCompletion()
else:
    pass  # TODO - eval
