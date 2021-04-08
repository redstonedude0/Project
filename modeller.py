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


def candidateSelection(dataset:Dataset,name="UNK",pad=True):
    # keep top 4 using p_e_m and top 3 using entity embeddings w/ context
    keep_pem = SETTINGS.n_cands_pem
    keep_context = SETTINGS.n_cands_ctx
    #Duplicates aren't allowed
    for doc in tqdm(dataset.documents, unit=name+"_documents", file=sys.stdout):
        for mention in doc.mentions:
            cands = mention.candidates
            unkcand = Candidate(-1,0,"#UNK#")
            # Sort p_e_m high to low
            cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
            #Trim to top 30 p_e_m, pad to 30 if padding
            if len(cands) > 30:
                # Need to trim to top 30 p_e_m
                cands = cands[0:30]  # Select top 30
            elif pad:#TODO - padding properly
                cands = cands + [unkcand]*(30-len(cands))#Pad to 30
            if SETTINGS.switches["switch_sel"]:#Changed from paper
                #Initially sort by embedding score
                cands.sort(key=lambda cand: _embeddingScore(mention, cand), reverse=True)
                #Then keep w.r.t ctx
                keptCands = cands[:keep_context]
                # Keep top w.r.t pem after
                cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
                for keptEmbeddingCand in cands:
                    if len(keptCands) == keep_context + keep_pem:
                        break  # NO MORE
                    # Don't add duplicates, unless that duplicate is the unknown candidate
                    if keptEmbeddingCand not in keptCands or keptEmbeddingCand == unkcand:
                        keptCands.append(keptEmbeddingCand)
                if len(keptCands) != keep_context + keep_pem:  # Should always be possible with unk_cand padding
                    raise RuntimeError(f"Incorrect number of candidates available ({len(keptCands)})")
                mention.candidates = keptCands
            else:
                keptCands = cands[:keep_pem]  # Keep top (keep_pem) always w.r.t PEM
                #NOTE: Paper does not allow duplicates
                # Keep top w.r.t ctx
                cands.sort(key=lambda cand: _embeddingScore(mention, cand), reverse=True)
                for keptEmbeddingCand in cands:
                    if len(keptCands) == keep_context + keep_pem:
                        break#NO MORE
                    #Don't add duplicates, unless that duplicate is the unknown candidate
                    if keptEmbeddingCand not in keptCands or keptEmbeddingCand == unkcand:
                        keptCands.append(keptEmbeddingCand)
                if len(keptCands) != keep_context + keep_pem:#Should always be possible with unk_cand padding
                    raise RuntimeError(f"Incorrect number of candidates available ({len(keptCands)})")
                mention.candidates = keptCands

def candidateSelection_full():
    candidateSelection(SETTINGS.dataset_train,"train",True)
    candidateSelection(SETTINGS.dataset_eval,"eval",True)

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
