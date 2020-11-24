# This file is for handling the model - it's resonsible for applying most hyperparameters and performing training

"""Do all the training on a specific dataset and neural model"""
import sys

from tqdm import tqdm

import neural
import processeddata
import utils
from datastructures import Model, Mention, Candidate
from hyperparameters import SETTINGS


def _embeddingScore(mention: Mention, candidate: Candidate):
    leftWords = mention.left_context.split(" ")
    rightWords = mention.right_context.split(" ")
    # TODO - assuming 50-word context window is 25 either side
    leftWords = leftWords[-25:]
    rightWords = rightWords[:25]
    wordSumVec = 0
    for word in leftWords + rightWords:
        wordSumVec += processeddata.wordid2embedding[processeddata.word2wordid.get(word, processeddata.unkwordid)]
    goldCand = mention.goldCand()
    if goldCand is not None:
        entityVec = processeddata.entid2embedding[processeddata.ent2entid[goldCand.text]]
        return entityVec.T.dot(wordSumVec)
    return 0


def candidateSelection():
    # TODO - select top 30 candidates per mention
    # TODO - keep top 4 p_e_m and top 3 et
    maxlen = 0
    for doc in tqdm(SETTINGS.dataset.documents, unit="documents", file=sys.stdout):
        for mention in doc.mentions:
            cands = mention.candidates
            # Sort p_e_m high to low
            cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
            if len(cands) > 30:
                # Need to trim to top 30 p_e_m
                cands = cands[0:30]  # Select top 30
            keptCands = cands[0:4]  # Keep top 4 always
            # TODO - assuming no duplicates appear, but duplicates take top 3 spots
            # Keep top 3 eT(Sigw)
            # TODO move some of this to GPU??
            cands.sort(key=lambda cand: _embeddingScore(mention, cand), reverse=True)
            keptEmbeddingCands = cands[0:3]
            for keptEmbeddingCand in keptEmbeddingCands:
                if keptEmbeddingCand not in keptCands:
                    keptCands.append(keptEmbeddingCand)
            mention.candidates = keptCands
            maxlen = max(maxlen, len(mention.candidates))
    print(maxlen)

def trainToCompletion():  # TODO - add params
    # TODO - checkpoint along the way
    utils.reportedRun("Candidate Selection", candidateSelection)

    model = Model()
    neural.train(model)
    # TODO - return EvaluationMetric object as well as final model?
    return model, None
