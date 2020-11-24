# This file obtains the processed/generated data (used by the model) and prepares it for use
# This processed data includes:
#   GloVe word embeddings (840B)
#   Generated entity embeddings

import numpy as np

from hyperparameters import SETTINGS

word2wordid = {}
ent2entid = {}
wordid2embedding = []
entid2embedding = []

'''Loading the entity and word embeddings and id maps'''


def loadEmbeddings():
    global word2wordid
    global ent2entid
    global wordid2embedding
    global entid2embedding
    with open(SETTINGS.dataDir_embeddings + "dict.word", "r") as f:
        for line in f:
            word2wordid[line] = len(word2wordid)
    with open(SETTINGS.dataDir_embeddings + "dict.entity", "r") as f:
        for line in f:
            # Normalise line - remove domain, make wiki-normalised according to wiki standard #TODO - link standard (I just happen to know it's _ => " " and remove%-encodings)
            line = line.replace("en.wikipedia.org/wiki/", "").replace("_", " ").replace("%22", '"')
            ent2entid[line] = len(ent2entid)

    wordid2embedding = np.load(SETTINGS.dataDir_embeddings + "word_embeddings.npy")
    entid2embedding = np.load(SETTINGS.dataDir_embeddings + "entity_embeddings.npy")
