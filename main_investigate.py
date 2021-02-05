# Main pipeline to run the paper

import datasets
import datastructures
import files
import modeller
import neural
import processeddata
from hyperparameters import APPLYBUNDLE_mentNorm, APPLYBUNDLE_paper
from utils import *

print("Cuda?", torch.cuda.is_available())
print("SETTINGS",SETTINGS)

reportedRun("Checking Datadir", files.checkDataDir)
reportedRun("Loading embeddings", processeddata.loadEmbeddings)
# Hook in via mount to foreign checkpoints
#SETTINGS.dataDir_checkpoints = "/home/harrison/Documents/project/mount/rds/user/hrjh2/hpc-work/checkpoints/"

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)

if SETTINGS.training:
    SETTINGS.dataset_train = datasets.loadDataset("aida_train.csv")
    SETTINGS.dataset_eval = datasets.loadDataset("aida_testA.csv")
#    SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:160]
    print(f"Size of training dataset: {len(SETTINGS.dataset_train.documents)}")
    print(f"Size of eval dataset: {len(SETTINGS.dataset_eval.documents)}")
    nextNum = 1
    for d in SETTINGS.dataset_train.documents:
        did = int(d.id.split(" ")[0])
        if did != nextNum:
            print(f"Expected {nextNum} but found {d.id}")
        nextNum = did + 1
    print(f"Last doc was {nextNum - 1}")
#    quit(0)
    # For debug SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:10]
    # load model
#    model = datastructures.Model.load("save_4")
#    model = datastructures.Model()
#    model.neuralNet = neural.NeuralNet()
#    model.evals = datastructures.EvalHistory()
#    for doc in SETTINGS.dataset.documents:
#        for m in doc.mentions:
#            if m.goldCandIndex() == -1:
#                print("Fail cand at doc",doc.id)
#    out = model.neuralNet(SETTINGS.dataset.documents[159])
#    print(f"Nans:{len(out[out != out])}")
#    SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:420]
    print(SETTINGS.dataset_train.documents[0].id)
    print(len(SETTINGS.dataset_train.documents))
    doc = SETTINGS.dataset_train.documents[251]
    print("DOC",len(doc.mentions))
    modeller.candidateSelection_full()
#    modeller.candidatePadding_full()
    print(len(doc.mentions[73].candidates))
    APPLYBUNDLE_mentNorm(SETTINGS)
    APPLYBUNDLE_paper(SETTINGS)
    SETTINGS.dataset_train.documents = SETTINGS.dataset_train.documents[0:50]
    model = modeller.trainToCompletion()

else:
    pass  # TODO - eval
