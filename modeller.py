# This file is for handling the model - it's resonsible for applying most hyperparameters and performing training

"""Do all the training on a specific dataset and neural model"""
import sys
import time

import torch
from tqdm import tqdm

import hyperparameters
import neural
import processeddata
import our_consistency
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

def _embeddingScore_paper(token_ids, entid):
    wordSumVec = 0
    for tokid in token_ids:
        wordSumVec += processeddata.wordid2embedding[tokid]
    wordSumVec /= len(token_ids)#mean not sum
    entityVec = processeddata.entid2embedding[entid]

    return entityVec.T.dot(wordSumVec)


def candidateSelection(dataset:Dataset,name="UNK",pad=True):
    tempNeural = neural.NeuralNet()
    # keep top 4 using p_e_m and top 3 using entity embeddings w/ context
    keep_pem = SETTINGS.n_cands_pem
    keep_context = SETTINGS.n_cands_ctx
    #Duplicates aren't allowed
    for doc in tqdm(dataset.documents, unit=name+"_documents", file=sys.stdout):
        l_contexts = [utils.stringToTokenIds(m.left_context,"left",SETTINGS.context_window_prerank) for m in doc.mentions]
        r_contexts = [utils.stringToTokenIds(m.right_context,"right",SETTINGS.context_window_prerank) for m in doc.mentions]
        token_idss = [l_ctx + r_ctx
                        if len(l_ctx) > 0 or len(r_ctx) > 0
                        else [processeddata.unkwordid]
                     for l_ctx, r_ctx in zip(l_contexts, r_contexts)]
        #flatten for consistency comparison
        flat_tokids = []
        flat_tokoffs = []
        for token_ids in token_idss:
            flat_tokoffs.append(len(flat_tokids))
            flat_tokids += token_ids
        our_consistency.save(torch.LongTensor(flat_tokids),"tokids")
        our_consistency.save(torch.LongTensor(flat_tokoffs),"tokoffs")
        if SETTINGS.switches["switch_sel"]:# Changed to match paper
            unkcand = Candidate(-1, 0, "#UNK#")
            #keep 30 w.r.t initial_prob (p_e_m scores)
            for mention, token_ids in zip(doc.mentions, token_idss):
                cands = mention.candidates
                # Sort p_e_m high to low
                cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
                # Trim to top 30 p_e_m, pad to 30 if padding
                if len(cands) > 30:
                    # Need to trim to top 30 p_e_m
                    cands = cands[0:30]  # Select top 30
                elif pad:  # TODO - padding properly
                    cands = cands + [unkcand] * (30 - len(cands))  # Pad to 30
                mention.candidates = cands
            all_entids = [
                [processeddata.ent2entid[cand.text] for cand in m.candidates]
                for m in doc.mentions
            ]
            our_consistency.save(torch.LongTensor(all_entids), "entids")
            #initially sort by embedding and keep top 4
            all_ents = [[processeddata.entid2embedding[entid] for entid in entids] for entids in all_entids]
            all_sents = []
            for token_ids in token_idss:
                wordSumVec = 0
                for tokid in token_ids:
                    wordSumVec += processeddata.wordid2embedding[tokid]
                wordSumVec /= len(token_ids)  # mean not sum
                all_sents.append(wordSumVec)
            #entity and context ('sentence') embeddings
            our_consistency.save(torch.FloatTensor(all_ents), "ntee_ents")
            our_consistency.save(torch.FloatTensor(all_sents), "ntee_sents")
            #embedding-entity scores
            probs = torch.FloatTensor(
                [[_embeddingScore_paper(token_ids, entid) for entid in entids] for entids, token_ids in
                 zip(all_entids, token_idss)])
            our_consistency.save(probs, "ntee_scores")
            #keep top keep_context
            # NOTE: https://github.com/pytorch/pytorch/issues/27542
            # Topk appears to be unstable between CPU and CUDA, among other differences - theirs gives lower indices, mine higher
            # When the values are equal.
            top_cand_vals, top_pos = torch.topk(probs, dim=1, k=keep_context)
            our_consistency.save(top_cand_vals, "top_pos_vals")  # TODO - normalize to match theirs
            our_consistency.save(top_pos.data.cpu().numpy(), "top_pos")  # Store as numpy cpu data, as the paper does
            top_cand_idss = torch.gather(torch.tensor(all_entids),dim=1,index=top_pos)
            top_cand_idss = top_cand_idss.tolist()
            #pad out based on p_e_m again
            #all_entids is already sorted by p_e_m
            for mention, top_cand_ids, ent_ids in zip(doc.mentions, top_cand_idss, all_entids):
                # Keep top w.r.t pem
                for ent_id in ent_ids:
                    if len(top_cand_ids) == keep_context + keep_pem:
                        break  # NO MORE
                    # Don't add duplicates, unless that duplicate is the unknown candidate
                    if ent_id not in top_cand_ids or ent_id == processeddata.unkentid:
                        top_cand_ids.append(ent_id)
                if len(top_cand_ids) != keep_context + keep_pem:  # Should always be possible with unk_cand padding
                    raise RuntimeError(f"Incorrect number of candidates available ({len(top_cand_ids)})")
                #cand ids back to cands
                cands = []
                for ent_id in top_cand_ids:
                    for c in mention.candidates:
                        if processeddata.ent2entid[c.text] == ent_id:
                            cands.append(c)
                            break #inner
                mention.candidates = cands

        else:
            for mention,token_ids in zip(doc.mentions,token_idss):
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
    #Append padding
    if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MentNorm and SETTINGS.switches["pad_enable"]:
        our_consistency.save(True, "use_pad_ent")
        # Padding entity
        #Text and candidates the neutral unknown entity?
        #  Cannot use text and cand as unk - some code relies on unk==no candidate, need special padding entity
        #  see paper source code for their implementation of this, they treat psi and phi differently for this, so we will too
        #pad_cand = Candidate(processeddata.unkentid, 1, "#UNK#")
        #pad_cands = [pad_cand for _ in range(0,keep_context+keep_pem)]
        #pad_mention = Mention.FromData(-1, "#UNK#", "", "", pad_cands, str(processeddata.unkentid))
        #for doc in dataset.documents:
        #    doc.mentions.append(pad_mention)
        # theirs:
        # add entity vec (randn(1)*0.1)
        # add mask (10000000)
        # add pem (10000000)
        # add psi! 0000000
        # increment n obviously
        # [entity vec fmc other (randn(1)*0.1)]
    else:
        our_consistency.save(False, "use_pad_ent")

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
