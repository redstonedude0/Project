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


def compute_embedding_score(mention: Mention, candidate: Candidate):
    ctx_window = 50#Size of the window context (EVEN)
    half_window = ctx_window//2
    left_words = mention.left_context.split(" ")
    right_words = mention.right_context.split(" ")
    left_words = left_words[-half_window:]
    right_words = right_words[:half_window]
    word_sum_vec = 0
    for word in left_words + right_words:
        word_sum_vec += processeddata.word_id_to_embedding[processeddata.word_to_word_id.get(word, processeddata.unk_word_id)]
    entity_vec = processeddata.ent_id_to_embedding[processeddata.ent_to_ent_id[candidate.text]]
    return entity_vec.T.dot(word_sum_vec)

def compute_embedding_score_paper(token_ids, ent_id):
    word_sum_vec = 0
    for tok_id in token_ids:
        word_sum_vec += processeddata.word_id_to_embedding[tok_id]
    word_sum_vec /= len(token_ids)#mean not sum
    entity_vec = processeddata.ent_id_to_embedding[ent_id]

    return entity_vec.T.dot(word_sum_vec)


def candidate_selection(dataset:Dataset, name="UNK", pad=True):
    # keep top 4 using p_e_m and top 3 using entity embeddingss w/ context
    keep_pem = SETTINGS.n_cands_pem
    keep_context = SETTINGS.n_cands_ctx
    #Duplicates aren't allowed
    for doc in tqdm(dataset.documents, unit=name+"_documents", file=sys.stdout):
        l_contexts = [utils.string_to_token_ids(m.left_context, "left", SETTINGS.context_window_prerank) for m in doc.mentions]
        r_contexts = [utils.string_to_token_ids(m.right_context, "right", SETTINGS.context_window_prerank) for m in doc.mentions]
        token_idss = [l_ctx + r_ctx
                        if len(l_ctx) > 0 or len(r_ctx) > 0
                        else [processeddata.unk_word_id]
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
            unk_cand = Candidate(-1, 0, "#UNK#")
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
                    cands = cands + [unk_cand] * (30 - len(cands))  # Pad to 30
                mention.candidates = cands
            all_ent_ids = [
                [processeddata.ent_to_ent_id[cand.text] for cand in m.candidates]
                for m in doc.mentions
            ]
            our_consistency.save(torch.LongTensor(all_ent_ids), "entids")
            #initially sort by embedding and keep top 4
            all_ents = [[processeddata.ent_id_to_embedding[entid] for entid in entids] for entids in all_ent_ids]
            all_sents = []
            for token_ids in token_idss:
                word_sum_vec = 0
                for tok_id in token_ids:
                    word_sum_vec += processeddata.word_id_to_embedding[tok_id]
                word_sum_vec /= len(token_ids)  # mean not sum
                all_sents.append(word_sum_vec)
            #entity and context ('sentence') embeddingss
            our_consistency.save(torch.FloatTensor(all_ents), "ntee_ents")
            our_consistency.save(torch.FloatTensor(all_sents), "ntee_sents")
            #embedding-entity scores
            probs = torch.FloatTensor(
                [[compute_embedding_score_paper(token_ids, entid) for entid in entids] for entids, token_ids in
                 zip(all_ent_ids, token_idss)])
            our_consistency.save(probs, "ntee_scores")
            #keep top keep_context
            # NOTE: https://github.com/pytorch/pytorch/issues/27542
            # Topk appears to be unstable between CPU and CUDA, among other differences - theirs gives lower indices, mine higher
            # When the values are equal.
            top_cand_vals, top_pos = torch.topk(probs, dim=1, k=keep_context)
            our_consistency.save(top_cand_vals, "top_pos_vals")  # TODO - normalize to match theirs
            our_consistency.save(top_pos.data.cpu().numpy(), "top_pos")  # Store as numpy cpu data, as the paper does
            top_cand_idss = torch.gather(torch.tensor(all_ent_ids),dim=1,index=top_pos)
            top_cand_idss = top_cand_idss.tolist()
            #pad out based on p_e_m again
            #all_ent_ids is already sorted by p_e_m
            for mention, top_cand_ids, ent_ids in zip(doc.mentions, top_cand_idss, all_ent_ids):
                # Keep top w.r.t pem
                for ent_id in ent_ids:
                    if len(top_cand_ids) == keep_context + keep_pem:
                        break  # NO MORE
                    # Don't add duplicates, unless that duplicate is the unknown candidate
                    if ent_id not in top_cand_ids or ent_id == processeddata.unk_ent_id:
                        top_cand_ids.append(ent_id)
                if len(top_cand_ids) != keep_context + keep_pem:  # Should always be possible with unk_cand padding
                    raise RuntimeError(f"Incorrect number of candidates available ({len(top_cand_ids)})")
                #cand ids back to cands
                cands = []
                for ent_id in top_cand_ids:
                    for c in mention.candidates:
                        if processeddata.ent_to_ent_id[c.text] == ent_id:
                            cands.append(c)
                            break #inner
                mention.candidates = cands

        else:
            for mention,token_ids in zip(doc.mentions,token_idss):
                cands = mention.candidates
                unk_cand = Candidate(-1,0,"#UNK#")
                # Sort p_e_m high to low
                cands.sort(key=lambda cand: cand.initial_prob, reverse=True)
                #Trim to top 30 p_e_m, pad to 30 if padding
                if len(cands) > 30:
                    # Need to trim to top 30 p_e_m
                    cands = cands[0:30]  # Select top 30
                elif pad:#TODO - padding properly
                    cands = cands + [unk_cand]*(30-len(cands))#Pad to 30
                kept_cands = cands[:keep_pem]  # Keep top (keep_pem) always w.r.t PEM
                #NOTE: Paper does not allow duplicates
                # Keep top w.r.t ctx
                cands.sort(key=lambda cand: compute_embedding_score(mention, cand), reverse=True)
                for kept_embedding_cand in cands:
                    if len(kept_cands) == keep_context + keep_pem:
                        break#NO MORE
                    #Don't add duplicates, unless that duplicate is the unknown candidate
                    if kept_embedding_cand not in kept_cands or kept_embedding_cand == unk_cand:
                        kept_cands.append(kept_embedding_cand)
                if len(kept_cands) != keep_context + keep_pem:#Should always be possible with unk_cand padding
                    raise RuntimeError(f"Incorrect number of candidates available ({len(kept_cands)})")
                mention.candidates = kept_cands
    #Append padding
    if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MENT_NORM and SETTINGS.switches["pad_enable"]:
        our_consistency.save(True, "use_pad_ent")
        # Padding entity
        #Text and candidates the neutral unknown entity?
        #  Cannot use text and cand as unk - some code relies on unk==no candidate, need special padding entity
        #  see paper source code for their implementation of this, they treat psi and phi differently for this, so we will too
        #pad_cand = Candidate(processeddata.unk_ent_id, 1, "#UNK#")
        #pad_cands = [pad_cand for _ in range(0,keep_context+keep_pem)]
        #pad_mention = Mention.from_data(-1, "#UNK#", "", "", pad_cands, str(processeddata.unk_ent_id))
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

def candidate_selection_full():
    candidate_selection(SETTINGS.dataset_train, "train", True)
    candidate_selection(SETTINGS.dataset_eval, "eval", True)

def train_to_completion():  # TODO - add params
    # TODO - checkpoint along the way
    print(f"Training on {len(SETTINGS.dataset_train.documents)} documents")
    print(f"Evaluating on {len(SETTINGS.dataset_eval.documents)} documents")
    cuda_avail = torch.cuda.is_available()
    print(f"Cuda? {cuda_avail}")
    if cuda_avail:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f"Using device {dev}")
    SETTINGS.device = device

    start_time = time.time()
    utils.reported_run("Candidate Selection", candidate_selection_full)

    model = Model()
    # Make the NN
    model_nn: neural.NeuralNet
    model_nn = neural.NeuralNet()
    model.neural_net = model_nn
    model.evals = EvalHistory()
    print("Neural net made, doing learning...")
    SETTINGS.DEBUG = False  # Prevent the model from spamming messages
    max_loops = 200
    max_no_improv = SETTINGS.learning_stop_threshold_epochs
    max_f1 = 0
    num_epochs_no_improv = 0
    lr = SETTINGS.learning_rate_initial
    for loop in range(0, max_loops):
        print(f"Doing loop {loop + 1}...")
        eval = neural.train(model,lr=lr)
        eval.step = loop + 1
        eval.time = time.time() - start_time
        eval.print()
        model.evals.metrics.append(eval)
        print(f"Loop {loop + 1} Done.")
        model.save(f"{SETTINGS.save_name}_{loop + 1}")
        if eval.accuracy >= SETTINGS.learning_reduction_threshold_f1:
            lr = SETTINGS.learning_rate_final
        if eval.accuracy > max_f1:
            max_f1 = eval.accuracy
            num_epochs_no_improv = 0
        else:
            num_epochs_no_improv += 1
        if num_epochs_no_improv >= max_no_improv:
            print(f"No improvement after {max_no_improv} loops. exiting")
            break
    # TODO - return EvaluationMetric object as well as final model?
    return model  # return final model
