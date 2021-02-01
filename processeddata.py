# This file obtains the processed/generated data (used by the model) and prepares it for use
# This processed data includes:
#   GloVe word embeddings (840B)
#   Generated entity embeddings

import numpy as np
from tqdm import tqdm


word2wordid = {}
unkwordid = 0
ent2entid = {}
unkentid = 0
wordid2embedding = []
entid2embedding = []

'''Loading the entity and word embeddings and id maps'''


def loadEmbeddings():
    from hyperparameters import SETTINGS
    global word2wordid
    global unkwordid
    global ent2entid
    global unkentid
    global wordid2embedding
    global entid2embedding
    if len(word2wordid) == 0:
        with open(SETTINGS.dataDir_embeddings + "dict.word", "r") as f:
            for line in tqdm(f, unit="words", total=492408):
                word = line.split("\t")[0]
                word2wordid[word] = len(word2wordid)
        with open(SETTINGS.dataDir_embeddings + "dict.entity", "r") as f:
            for line in tqdm(f, unit="entities", total=274474):
                entname = line.split("\t")[0]
                # Normalise entname - remove domain, make wiki-normalised according to wiki standard #TODO - link standard (I just happen to know it's _ => " " and remove%-encodings)
                entname = entname.replace("en.wikipedia.org/wiki/", "").replace("_", " ").replace("%22", '"')
                ent2entid[entname] = len(ent2entid)
        unkwordid = word2wordid["#UNK#"]
        unkentid = len(ent2entid)
        ent2entid["#UNK#"] = unkentid
        wordid2embedding = np.load(SETTINGS.dataDir_embeddings + "word_embeddings.npy")
        entid2embedding = np.load(SETTINGS.dataDir_embeddings + "entity_embeddings.npy")
        unkentembed = np.mean(entid2embedding, axis=0, keepdims=True)  # mean embedding, as a (1,d) array
        entid2embedding = np.append(entid2embedding, unkentembed, axis=0)  # append as 1D
    else:
        print("ALREADY LOADED EMBEDDINGS!")
