# Main pipeline to run the paper

import datasets
import datastructures
import files
import modeller
import neural
import processeddata
from hyperparameters import apply_bundle_ment_norm, apply_bundle_paper, apply_bundle_rel_norm
from utils import *

print("Cuda?", torch.cuda.is_available())
print("SETTINGS",SETTINGS)

reported_run("Checking Datadir", files.check_data_dir)
reported_run("Loading embeddingss", processeddata.load_embeddings)
# Hook in via mount to foreign checkpoints
#SETTINGS.data_dir_checkpoints = "/home/harrison/Documents/project/mount/rds/user/hrjh2/hpc-work/checkpoints/"

# TODO - Input parameters, specify data & checkpoint locations, etc
# TODO - should these be params or runtime selections?

# TODO - Specify what to do (train, eval, etc)
# TODO - perform action (obtain dataset, use/make neural network as required, save results as specified)

if SETTINGS.training:
    SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv")
    SETTINGS.dataset_eval = datasets.load_dataset("aida_testA.csv")
#    SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:160]
    print(f"Size of training dataset: {len(SETTINGS.dataset_train.documents)}")
    print(f"Size of eval dataset: {len(SETTINGS.dataset_eval.documents)}")
    next_num = 1
    for d in SETTINGS.dataset_train.documents:
        did = int(d.id.split(" ")[0])
        if did != next_num:
            print(f"Expected {next_num} but found {d.id}")
        next_num = did + 1
    print(f"Last doc was {next_num - 1}")
#    quit(0)
    # For debug SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:10]
    # load model
#    model = datastructures.Model.load("save_4")
#    model = datastructures.Model()
#    model.neural_net = neural.NeuralNet()
#    model.evals = datastructures.EvalHistory()
#    for doc in SETTINGS.dataset.documents:
#        for m in doc.mentions:
#            if m.goldCandIndex() == -1:
#                print("Fail cand at doc",doc.id)
#    out = model.neural_net(SETTINGS.dataset.documents[159])
#    print(f"Nans:{len(out[out != out])}")
#    SETTINGS.dataset.documents = SETTINGS.dataset.documents[0:420]
    print(SETTINGS.dataset_train.documents[0].id)
    print(len(SETTINGS.dataset_train.documents))
    doc = SETTINGS.dataset_train.documents[251]
    print("DOC",len(doc.mentions))
    modeller.candidate_selection_full()
#    modeller.candidatePadding_full()
    print(len(doc.mentions[73].candidates))
    apply_bundle_rel_norm(SETTINGS)
    apply_bundle_paper(SETTINGS)
    SETTINGS.dataset_train.documents = SETTINGS.dataset_train.documents[0:50]
    model = modeller.train_to_completion()

else:
    pass  # TODO - eval
