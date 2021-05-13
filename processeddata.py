# This file obtains the processed/generated data (used by the model) and prepares it for use
# This processed data includes:
#   GloVe word embeddingss (840B)
#   Generated entity embeddingss

import numpy as np
from tqdm import tqdm


#TODO - word2id should only be accessible via utils.normalise
word_to_word_id = {}
unk_word_id = 0
ent_to_ent_id = {}
unk_ent_id = 0
word_id_to_embedding = []
ent_id_to_embedding = []

word_to_word_id_snd = {}
unk_word_id_snd = 0
word_id_to_embedding_snd = []

'''Loading the entity and word embeddingss and id maps'''


def load_embeddings():
    from hyperparameters import SETTINGS
    global word_to_word_id
    global unk_word_id
    global ent_to_ent_id
    global unk_ent_id
    global word_id_to_embedding
    global ent_id_to_embedding
    global word_to_word_id_snd
    global unk_word_id_snd
    global word_id_to_embedding_snd
    if len(word_to_word_id) == 0:
        with open(SETTINGS.data_dir_embeddings + "dict.word", "r") as f:
            for idx,line in tqdm(enumerate(f), unit="words", total=492408):
                word = line.split("\t")[0]
                if word in word_to_word_id:
                    print("Duplicate word ", word)
                word_to_word_id[word] = idx

        with open(SETTINGS.data_dir_embeddings + "dict.entity", "r") as f:
            for idx,line in tqdm(enumerate(f), unit="entities", total=274474):
                ent_name = line.split("\t")[0]
                # Normalise ent_name - remove domain, make wiki-normalised according to wiki standard #TODO - link standard (I just happen to know it's _ => " " and remove%-encodings)
                ent_name = ent_name.replace("en.wikipedia.org/wiki/", "").replace("_", " ").replace("%22", '"')
                if ent_name in ent_to_ent_id:
                    print("Duplicate ent_name ", ent_name)
                ent_to_ent_id[ent_name] = idx
        unk_word_id = word_to_word_id["#UNK#"]
        unk_ent_id = len(ent_to_ent_id)
        ent_to_ent_id["#UNK#"] = unk_ent_id
        word_id_to_embedding = np.load(SETTINGS.data_dir_embeddings + "word_embeddings.npy")
        ent_id_to_embedding = np.load(SETTINGS.data_dir_embeddings + "entity_embeddings.npy")
        unk_ent_embed = np.mean(ent_id_to_embedding, axis=0, keepdims=True)  # mean embedding, as a (1,d) array

        #Adjust following paper
        if SETTINGS.switches["override_embs"]:#Normalisation from paper
            #Normalise each embedding by dividing by its length, maxed at 1e-12 to prevent div0, with unkemb being set to all 1e-10
            max_embs = np.maximum(np.linalg.norm(ent_id_to_embedding,
                                                axis=1, keepdims=True), 1e-12)
            ent_id_to_embedding /= max_embs
            unk_ent_embed[:,:] = 1e-10
            max_embs = np.maximum(np.linalg.norm(word_id_to_embedding,
                                                axis=1, keepdims=True), 1e-12)
            word_id_to_embedding /= max_embs#
            word_id_to_embedding[unk_word_id][:] = 1e-10

        ent_id_to_embedding = np.append(ent_id_to_embedding, unk_ent_embed, axis=0)  # append as 1D

        if SETTINGS.switches["snd_embs"]:
            #Create second embeddingss too
            #TODO - do embeddingss need required_grad False to be manually set?
            print("strt", len(word_to_word_id_snd))
            with open(SETTINGS.data_dir_embeddings + "glove/dict.word", "r") as f:
                for idx,line in tqdm(enumerate(f), unit="words_snd", total=1):
                    word = line.split("\t")[0]
                    if word in word_to_word_id_snd:
                        print("Duplicate snd word ",word)
                    word_to_word_id_snd[word] = idx
            unk_word_id_snd = word_to_word_id_snd["#UNK#"]
            word_id_to_embedding_snd = np.load(SETTINGS.data_dir_embeddings + "glove/word_embeddings.npy")
            #no embedding normalisation on these?

    else:
        print("ALREADY LOADED EMBEDDINGS!")
