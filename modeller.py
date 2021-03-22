# This file is for handling the model - it's resonsible for applying most hyperparameters and performing training

"""Do all the training on a specific dataset and neural model"""
import sys
import time

import torch
from tqdm import tqdm

import neural
import processeddata
import utils
from datastructures import Model, Mention, Candidate, EvalHistory, Dataset
from hyperparameters import SETTINGS


def _embeddingScore(mention: Mention, candidate: Candidate):
    ctxwindow = 50#Size of the window context (EVEN)
    halfwindow = ctxwindow//2
    leftWords = mention.left_context.split(" ")
    rightWords = mention.right_context.split(" ")
    leftWords = leftWords[-halfwindow:]
    rightWords = rightWords[:halfwindow]
    wordSumVec = 0
    for word in leftWords + rightWords:
        wordSumVec += processeddata.wordid2embedding[processeddata.word2wordid.get(word, processeddata.unkwordid)]
    entityVec = processeddata.entid2embedding[processeddata.ent2entid[candidate.text]]
    return entityVec.T.dot(wordSumVec)


def candidateSelection(dataset:Dataset,name="UNK"):
    # keep top 4 using p_e_m and top 3 using entity embeddings w/ context
    keep_pem = 4#As prescribed by paper
    keep_context = 3#Paper says 3 but code uses 4
    pad = True#Pad with UNKs
    #Duplicates aren't allowed
    for doc in tqdm(dataset.documents, unit=name+"_documents", file=sys.stdout):
        for mention in doc.mentions:
            cands = mention.candidates
            unkcand = Candidate(-1,0,"#UNK#")
            # Sort p_e_m high to low
            cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
            if len(cands) > 30:
                # Need to trim to top 30 p_e_m
                cands = cands[0:30]  # Select top 30
            elif pad:
                cands = cands + [unkcand]*(30-len(cands))#Pad to 30
            keptCands = cands[0:keep_pem]  # Keep top (4) always
            # TODO - assuming no duplicates appear, but duplicates take top 3 spots
            # Keep top 3 eT(SUM w)
            # TODO move some of this to GPU??
            # TODO can't work out how paper does this - they appear to take the top 7 based on embedding context and ignore p_e_m??? INVESTIGATE
            cands.sort(key=lambda cand: _embeddingScore(mention, cand), reverse=True)
            for keptEmbeddingCand in cands:
                if len(keptCands) == keep_context + keep_pem:
                    break#NO MORE
                if keptEmbeddingCand not in keptCands:
                    keptCands.append(keptEmbeddingCand)
            if len(keptCands) != keep_context + keep_pem:
                pass
#                raise RuntimeError(f"Incorrect number of candidates available ({len(keptCands)})")
            #TODO - finish? I think I need to pad candidates also
            mention.candidates = keptCands

def candidateSelection_full():
    candidateSelection(SETTINGS.dataset_train,"train")
    candidateSelection(SETTINGS.dataset_eval,"eval")

def candidatePadding(dataset:Dataset,name="UNK"):
    # make sure always 7 candidates
    paddingCand = Candidate(-1,0,"#UNK#")
    for doc in tqdm(dataset.documents, unit=name+"_documents", file=sys.stdout):
        for mention in doc.mentions:
            cands = mention.candidates
            if len(cands) < 7:
                paddingCands = [paddingCand for _ in range(len(cands),7)]
                mention.candidates += paddingCands

def candidatePadding_full():
    candidatePadding(SETTINGS.dataset_train,"train")
    candidatePadding(SETTINGS.dataset_eval,"eval")

def trainToCompletion():  # TODO - add params
    # TODO - checkpoint along the way
    print(f"Training on {len(SETTINGS.dataset_train.documents)} documents")
    print(f"Evaluating on {len(SETTINGS.dataset_eval.documents)} documents")
    cudaAvail = torch.cuda.is_available()
    print(f"Cuda? {cudaAvail}")
    if cudaAvail:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"Using device {dev}")
    SETTINGS.device = device

    startTime = time.time()
    utils.reportedRun("Candidate Selection", candidateSelection_full)
    model = Model()
    # Make the NN
    model_nn: neural.NeuralNet
    model_nn = neural.NeuralNet()
    model.neuralNet = model_nn
    model.evals = EvalHistory()
    print("Neural net made, doing learning...")
    SETTINGS.DEBUG = False  # Prevent the model from spamming messages
    maxLoops = 50
    maxNoImprov = 20
    maxF1 = 0
    numEpochsNoImprov = 0
    lr = SETTINGS.learning_rate_initial
    for loop in range(0, maxLoops):
        print(f"Doing loop {loop + 1}...")
        eval = neural.train(model,lr=lr)
        eval.step = loop + 1
        eval.time = time.time() - startTime
        eval.print()
        model.evals.metrics.append(eval)
        print(f"Loop {loop + 1} Done.")
        model.save(f"{SETTINGS.saveName}_{loop + 1}")
        if eval.accuracy >= SETTINGS.learning_reduction_threshold_f1:
            lr = SETTINGS.learning_rate_final
        if eval.accuracy > maxF1:
            maxF1 = eval.accuracy
            numEpochsNoImprov = 0
        else:
            numEpochsNoImprov += 1
        if numEpochsNoImprov >= maxNoImprov:
            print(f"No improvement after {maxNoImprov} loops. exiting")
            break
    # TODO - return EvaluationMetric object as well as final model?
    return model  # return final model
