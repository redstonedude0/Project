# This file obtains the training and evaluation datasets and prepares it for use.
#
# Note: Data provided has been pre-computed with p_e_m probabilities
# Data is from sources:
# training set:
#   AIDA-CoNLL>AIDA-train - 946 docs
# dev evaluation set:
#   AIDA-CoNLL>AIDA-A - 216 docs
# test evaluation sets:
#   AIDA-CoNLL>AIDA-B - 231 docs
#   MSNBC - 20 docs
#   AQUAINT - 50 docs
#   ACE2004 - 36 docs
#   WNED-CWEB(CWEB) - 320 docs
#   WNED-WIKI(WIKI) - 320 docs
#
# AIDA sets are considered in-domain, other sets are considered out-domain.
#

from enum import Enum, auto

from datastructures import Dataset, Mention, Candidate, Document


class DatasetType(Enum):
    TRAIN = auto()  # training
    TEST = auto()  # evaluation
    DEV = auto()  # evaluation

from hyperparameters import SETTINGS


def loadDataset(csvPath: str):
    csvPath = SETTINGS.dataDir_csv + csvPath
    dataset = Dataset()
    dataset.documents = []
    with open(csvPath, "r") as f:
        # Iterate over CSV structure - each line is a mention, when the ID changes the doc changes
        doc = Document()
        for line in f:
            mention = Mention()
            parts = line.split("\t")
            doc_id1 = parts[0]
            doc_id2 = parts[1]
            if doc_id1 != doc_id2:
                # As I understand it the 1st 2 columns are equal, raise error if this invariant is broken
                raise NotImplementedError(f"Document ids not equal {doc_id1} {doc_id2}")
            if doc_id1 != doc.id:
                # New doc
                if doc.id != None:  # None if initial line
                    dataset.documents.append(doc)
                doc = Document()
                doc.mentions = []
            doc.id = doc_id1  # make sure always set
            # Actually set up mention
            mention.text = parts[2]
            mention.left_context = parts[3]
            mention.right_context = parts[4]
            if parts[5] != "CANDIDATES":
                # As I understand it this is always equal, raise error if this invariant is broken
                raise NotImplementedError(f"Col5 not CANDIDATES on {doc_id1}")
            candidates = [cand for cand in parts[6:-2]]
            if len(candidates) == 1 and candidates[0] == "EMPTYCAND":
                candidates = []
            candidates = [cand.split(",") for cand in candidates]
            candidates = [(cand[0], cand[1], cand[2:]) for cand in candidates]  # ERRORS
            candidates = [Candidate(id, prob, ",".join(nameparts)) for (id, prob, nameparts) in candidates]
            mention.candidates = candidates
            mention.gold_id = -1  # no id by default
            if parts[-2] != "GT:":
                # As I understand it this is always equal, raise error if this invariant is broken
                raise NotImplementedError(f"Col-2 not GT on {doc_id1}")
            goldDataParts = parts[-1].split(",")
            if len(goldDataParts) != 1:  # otherwise -1 anyway
                mention.gold_id = goldDataParts[1]  # the ID of the candidate
            doc.mentions.append(mention)
    pass  # TODO

# Removed - datasets can be loaded as necessary
#    loadDataset(train_AIDA, "aida_train.csv")
#    loadDataset(dev_AIDA, "aida_testA.csv")
#    loadDataset(test_AIDA, "aida_testB.csv")
#    loadDataset(test_MSNBC, "wned-msnbc.csv")
#    loadDataset(test_AQUAINT, "wned-aquaint.csv")
#    loadDataset(test_ACE2004, "wned-ace2004.csv")
#    loadDataset(test_CWEB, "wned-clueweb.csv")
#    loadDataset(test_WIKI, "wned-wikipedia.csv")
