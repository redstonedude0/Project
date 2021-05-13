# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import hyperparameters
import our_consistency
import processeddata
import utils
from datastructures import Model, Document, EvaluationMetrics, Candidate, Mention
from hyperparameters import SETTINGS
from utils import debug, map_2D, map_1D, smart_sum, smart_max, set_mask_broadcastable, \
    normalise_avg_to_zero_rows, nan_test


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model, takes a learning rate'''


def train(model: Model, lr=SETTINGS.learning_rate_initial):
    # Prepare for training
    if SETTINGS.allow_nans:
        raise Exception("Fatal error - cannot learn with allow_nans enabled")
    model.neural_net.to(SETTINGS.device)  # Move to correct device
    model.neural_net.train()  # Set training flag #TODO - why?
    torch.autograd.set_detect_anomaly(True)  # True for debugging to detect gradient anomolies

    # Initialise optimizer, calculate loss
    optimizer = torch.optim.Adam(model.neural_net.parameters(), lr=lr)
    optimizer.zero_grad()  # zero all gradients to clear buffers
    total_loss = 0
    # loss = loss_regularisation(model.neural_net.R, model.neural_net.D)
    # TODO - check - does the original paper add this regularisation once per loop?
    #    loss.backward()
    # total_loss += loss.item()
    eval_correct = 0
    eval_wrong = 0
    true_pos = 0  # guessed right valid index when valid index possible
    false_pos = 0  # guessed wrong index when valid index possible?
    false_neg = 0  # guessed wrong index when valid index possible?
    possible_correct = 0
    total = 0

    their_eval_truepos = 0
    their_eval_total_non_nil_predictions = 0
    their_eval_total = 0
    for doc_idx, document in enumerate(tqdm(SETTINGS.dataset_train.documents, unit="train_documents", file=sys.stdout)):
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MENT_NORM and SETTINGS.switches["pad_enable"]:
            # backup mentions
            backup_mentions = document.mentions.copy()
        if SETTINGS.low_mem:
            if len(document.mentions) > 200:
                print(
                    f"Unable to learn on document {doc_idx} ({len(document.mentions)} mentions would exceed memory limits)")
                continue
        try:
            out = model.neural_net(document)
        except RuntimeError as err:
            if "can't allocate memory" in err.__str__():
                print(f"Memory allocation error on {doc_idx} - {len(document.mentions)} mentions exceeds memory limits")
                continue  # next iteration of loop
            else:
                raise err
        if len(out[out != out]) >= 1:
            # Dump
            print("Document index", doc_idx)
            print("Document id", document.id)
            print("Model output", out)
            print("Found nans in model output! Cannot proceed with learning", file=sys.stderr)
            continue  # next loop of document
        loss = loss_document(document, out)
        loss += loss_regularisation(model.neural_net.R_diag, model.neural_net.D_diag)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        # Calculate evaluation metric data
        truth_indices = torch.tensor([m.gold_cand_index() for m in document.mentions]).to(SETTINGS.device)
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        best_cand_indices = out.max(dim=1)[1]  # index of maximum across candidates #1D(n) tensor (not -1)
        same_list = truth_indices.eq(best_cand_indices)
        correct = same_list.sum().item()  # extract value from 0-dim tensor
        wrong = (~same_list).sum().item()
        eval_correct += correct
        eval_wrong += wrong
        # TODO - paper adds #UNK#/NIL cands to pad - this should not be a tp/fp
        true_pos += correct  # where the same, we have TP
        got_wrong = ~same_list  # ones we got wrong
        possible = truth_indices != -1  # ones possible to get right
        missed = got_wrong.logical_and(possible)  # ones we got wrong but could've got right
        # NOTE - we will have no TP for #UNK# (-1 gold) mentions
        false_neg += missed.sum().item()  # NOTE:sum FP=FN in the micro-averaging case
        false_pos += missed.sum().item()
        possible_correct += possible.sum().item()
        total += len(same_list)
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MENT_NORM and SETTINGS.switches["pad_enable"]:
            # unbackup mentions
            document.mentions = backup_mentions

    if SETTINGS.loss_patched:
        eval_correct = 0
        eval_wrong = 0
        true_pos = 0
        false_pos = 0
        false_neg = 0
        possible_correct = 0
        total = 0
        # Do it their way
        for doc_idx, document in enumerate(
                tqdm(SETTINGS.dataset_eval.documents, unit="eval_documents", file=sys.stdout)):
            if SETTINGS.low_mem:
                if len(document.mentions) > 200:
                    print(
                        f"Unable to eval on document {doc_idx} ({len(document.mentions)} mentions would exceed memory limits)")
                    continue
            try:
                out = model.neural_net(document)
            except RuntimeError as err:
                if "can't allocate memory" in err.__str__():
                    print(
                        f"Memory allocation error on {doc_idx} - {len(document.mentions)} mentions exceeds memory limits")
                    continue  # next iteration of loop
                else:
                    raise err
            if len(out[out != out]) >= 1:
                # Dump
                print("Document index", doc_idx)
                print("Document id", document.id)
                print("Model output", out)
                print("Found nans in model output! Cannot proceed with eval", file=sys.stderr)
                continue  # next loop of document
            # Calculate evaluation metric data
            truth_indices = torch.tensor([m.gold_cand_index() for m in document.mentions]).to(SETTINGS.device)
            # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
            best_cand_indices = out.max(dim=1)[1]  # index of maximum across candidates #1D(n) tensor (not -1)
            same_list = truth_indices.eq(best_cand_indices)
            correct = same_list.sum().item()  # extract value from 0-dim tensor
            wrong = (~same_list).sum().item()
            eval_correct += correct
            eval_wrong += wrong
            # TODO - paper adds #UNK#/NIL cands to pad - this should not be a tp/fp
            true_pos += correct  # where the same, we have TP
            got_wrong = ~same_list  # ones we got wrong
            possible = truth_indices != -1  # ones possible to get right
            missed = got_wrong.logical_and(possible)  # ones we got wrong but could've got right
            # NOTE - we will have no TP for #UNK# (-1 gold) mentions
            false_neg += missed.sum().item()  # NOTE:sum FP=FN in the micro-averaging case
            false_pos += missed.sum().item()
            possible_correct += possible.sum().item()
            total += len(same_list)

            masks = [[True] * len(m.candidates) + [False] * (SETTINGS.n_cands - len(m.candidates)) for m in
                     document.mentions]
            masks = torch.tensor(masks).to(SETTINGS.device)
            # Where the best candidate isn't real, try guessing 0
            best_cand_masks = masks.T[best_cand_indices].diagonal()
            best_cand_indices[~best_cand_masks] = 0
            # Where the best candidate still isn't real, guess -1
            best_cand_masks = masks.T[best_cand_indices].diagonal()
            best_cand_indices[~best_cand_masks] = -1
            same_list = truth_indices.eq(best_cand_indices)
            same_list[best_cand_indices == -1] = False  # If Nil then not "the same"
            their_eval_truepos += same_list.sum().item()
            their_eval_total_non_nil_predictions += (best_cand_indices >= 0).sum().item()
            their_eval_total += len(same_list)

    # Learn!
    print("Stepping...")
    #    optimizer.step()  # step - update parameters backwards as requried
    print("Done.")

    # Evaluate
    #    microPrecision = true_pos/(true_pos+false_pos)    #Equal to microF1 in this case
    eval = EvaluationMetrics()
    eval.loss = total_loss
    eval.accuracy = eval_correct / (eval_correct + eval_wrong)
    if SETTINGS.loss_patched:
        precision = their_eval_truepos / their_eval_total_non_nil_predictions
        recall = their_eval_truepos / their_eval_total
        their_accuracy = 2 * precision * recall / (precision + recall)
        print("precision", precision)
        print("recall", recall)
        print(eval.accuracy, "against", their_accuracy)
        eval.accuracy = their_accuracy
    eval.accuracy_possible = possible_correct / total
    #    eval.correctRatio =
    #    eval.microF1 = microPrecision
    #    eval.correctRatio_possible = eval_correct / possible_correct
    #    eval.microF1_possible = true_pos/possible_correct
    return eval


def loss_document(document: Document, output):
    if SETTINGS.loss_patched:
        truth_indices = torch.tensor([m.gold_cand_index() for m in document.mentions]).to(SETTINGS.device)
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        truth_indices[truth_indices == -1] = 0
        p_i_e = output  # 2D (n,n_cands) tensor of p_i_e values
        import torch.nn.functional as F
        loss = F.multi_margin_loss(p_i_e, truth_indices, margin=SETTINGS.gamma)
        return loss

    else:
        n = len(document.mentions)
        # MRN equation 7
        p_i_e = output  # 2D (n,n_cands) tensor of p_i_e values
        truth_indices = [m.gold_cand_index() for m in document.mentions]
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        p_i_e_star = p_i_e.transpose(0, 1)[truth_indices].diagonal()
        # TODO - need to set piestar to 0 where truthindex -1?
        # broadcast (n,1) to (n,n_cands)
        gamma_eqn = SETTINGS.gamma - p_i_e_star.reshape([n, 1]) + p_i_e
        # Max each element with 0 (repeated to (n,n_cands))
        h = torch.max(gamma_eqn, torch.tensor(0.).to(SETTINGS.device).repeat([n, SETTINGS.n_cands]))
        l = smart_sum(h)  # ignore nans in loss function (outside of candidate range)
        return l


def loss_regularisation(R_diag, D_diag):
    R = R_diag.diag_embed()
    D = D_diag.diag_embed()
    regularisation_R = 0
    for i in range(0, SETTINGS.k):
        for j in range(0, i):
            # Do for every ordered pair i,j
            # If the paper means to do for all possible combinations i,j ignoring order then change this
            # TODO
            regularisation_R += dist(R[i], R[j])
    regularisation_D = 0
    for i in range(0, SETTINGS.k):
        for j in range(0, i):
            # TODO Change if necesssary
            regularisation_D += dist(D[i], D[j])
    regularisation_R *= SETTINGS.lambda1
    regularisation_D *= SETTINGS.lambda2
    regularisation = regularisation_R + regularisation_D
    return regularisation


def dist(x, y):
    # p-norm with p=2 is the Euclidean/Frobenius norm given by .norm()
    x_part = x / x.norm()
    y_part = y / y.norm()
    return (x_part - y_part).norm()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        torch.manual_seed(SETTINGS.SEED)
        self.f = nn.Sequential(
            nn.MultiheadAttention(300, 1, SETTINGS.dropout_rate)
            # TODO - absolutely no idea how to do this, this is a filler for now
        )  # Attention mechanism to achieve feature representation
        torch.manual_seed(SETTINGS.SEED)
        self.f_m_c = nn.Sequential(
            nn.Linear(900, 300),  # TODO what dimensions?
            nn.Tanh(),
            nn.Dropout(p=SETTINGS.dropout_rate),
        )
        our_consistency.save(list(self.f_m_c.parameters()), "fmc_initialweight")
        g_hid_dims = 100  # TODO - this is the default used by the paper's code, not mentioned anywhere in paper
        self.g = nn.Sequential(
            nn.Linear(2, g_hid_dims),
            nn.ReLU(),
            nn.Linear(g_hid_dims, 1)
            # TODO - this is the 2-layer nn 'g' referred to in the paper, called score_combine in mulrel_ranker.py
        )
        B_diag1 = torch.ones(300).to(SETTINGS.device)
        B_diag2 = torch.ones(300).to(SETTINGS.device)
        # randn~Norm(0,1)
        R_diag = torch.randn(SETTINGS.k, 300).to(SETTINGS.device) * 0.1
        torch.manual_seed(SETTINGS.SEED)
        D_diag = torch.randn(SETTINGS.k, 300).to(SETTINGS.device) * 0.1
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MENT_NORM and SETTINGS.switches["exp_adjust"]:
            torch.manual_seed(SETTINGS.SEED)
            D_diag = torch.randn(SETTINGS.k, 300).to(SETTINGS.device) * 0.01

        if not SETTINGS.rel_specialinit:
            R_diag += 1

        self.register_parameter("B_diag", torch.nn.Parameter(B_diag))
        self.register_parameter("B_diag1", torch.nn.Parameter(B_diag1))
        self.register_parameter("B_diag2", torch.nn.Parameter(B_diag2))
        self.register_parameter("R_diag", torch.nn.Parameter(R_diag))
        self.register_parameter("D_diag", torch.nn.Parameter(D_diag))
        # Default (normally sampled) entity and context
        torch.manual_seed(SETTINGS.SEED)
        self.register_parameter("pad_ent", torch.nn.Parameter(torch.randn(300).to(SETTINGS.device) * 0.1))
        self.register_parameter("pad_ctx", torch.nn.Parameter(torch.randn(300).to(SETTINGS.device) * 0.1))
        # TODO - check the distribution of D, paper seems to be different than impl? (0.01ment/0.1rel)
        # TODO - check for B, paper seems to have 2 matrices to do ctx score? (ctxattrranker.forward)

    # local and pairwise score functions (equation 3+section 3.1)

    '''
    Compute CTX embedding f(c)
    INPUT:
    n: len(mentions)
    embeddingss: 3D (n,n_cands,d) tensor of embeddingss of entities
    token_embeddingss: 3D (n,window_size,d) tensor of embeddingss of tokens in window around each mention
    token_maskss: 3D (n,window_size) bool tensor of masks
    RETURN:
    3D tensor (n,n_cands,d) of context embeddingss
    '''

    def f_c(self, n, embeddingss, token_embeddingss, token_maskss):
        window_size = token_maskss.shape[1]  # Window size is dynamic, up to 100
        # For each mention i, we compute the correlation for each token t, for each candidate (of i) c
        # embeddingss (n,n_cands,d) * diag weighting.diag() (d,d) * token_embeddingss (n,win,d)
        weighted_token_embeddingss = torch.matmul(self.B_diag2.diag_embed(),
                                                  token_embeddingss.transpose(1, 2))  # (n,d,win)
        token_entity_scoresss = torch.matmul(embeddingss, weighted_token_embeddingss)  # (n,n_cands,win)
        # set to -1e10 for unknown tokens
        token_entity_scoresss[~token_maskss.reshape([n, 1, window_size]).repeat([1, SETTINGS.n_cands, 1])] = -1e10

        # Let the score of each token be the best score across all candidates (ignore indexes)
        token_scoress, _ = torch.max(token_entity_scoresss, dim=1)  # (n,win)

        # Take the top (default 25) scores (unsorted!), with corresponding id
        best_scoress, best_scoress_idx = torch.topk(token_scoress, dim=1, k=min(SETTINGS.attention_token_count,
                                                                                window_size))  # (n,25), (n,25) [technically <25 if small window]

        # Take the actual embeddingss for the top 25 tokens
        best_tokenss = torch.gather(token_embeddingss, dim=1,
                                    index=best_scoress_idx.view(n, -1, 1).repeat(1, 1, SETTINGS.d))  # (n,25,d)

        # Scale each tokens embedding by prob
        token_probss = nn.functional.softmax(best_scoress, dim=1)  # (n,25)
        token_probss = token_probss / torch.sum(token_probss, dim=1,
                                                keepdim=True)  # (n,25), linear normalise because Le et al. does

        best_tokenss = best_tokenss * token_probss.view(n, -1, 1)  # multiplication will broadcast, (n,25,d)

        # Sum the 25-best window to achieve a context embedding weighted by token probability
        context_embeddings = torch.sum(best_tokenss, dim=1)  # (n,d)

        return context_embeddings

    '''
    Compute PSI for all candidates for all mentions
    INPUT:
    n: len(mentions)
    embeddingss: 3D (n,n_cands,d) tensor of embeddingss
    masks: 2D (n,n_cands) bool tensor mask
    token_embeddingss: 3D tensor(n,win,d) tensor of embeddingss of tokens in window around each mention
    tokenMaskss: 3D bool tensor(n,win) tensor of masks
    RETURN:
    2D tensor (n,n_cands) psi values per candidate per mention
    '''

    def psiss(self, n, embeddings, masks, tokenEmbeddings, tokenMaskss):
        # Compute context embeddingss f(c)
        fcs = self.f_c(n, embeddings, tokenEmbeddings, tokenMaskss)  # (n,d)
        # embeddingss is 3D (n,n_cands,d) tensor
        # B is 2D (d,d) tensor (from (d) B_diag tensor)

        # embeddingss (n,n_cands,d) * diag weighting.diag() (d,d) * fcs (n,d)
        weighted_context_embeddings = torch.matmul(self.B_diag1.diag_embed(), fcs.T).T  # (n,d)
        weighted_context_embeddings = weighted_context_embeddings.view(n, SETTINGS.d, 1)  # (n,d,1)
        valss = torch.matmul(embeddings, weighted_context_embeddings)  # (n,n_cands,1)

        # remove extra dim
        valss = valss.view(n, SETTINGS.n_cands)
        # vals2 is 2D (n,n_cands) tensor

        valss[~masks] = -1e10

        if SETTINGS.switches["consistency_psi"]:
            # TODO - add consistency checks
            our_consistency.save(None, "psi_i_pem")  # hardcoded but check anyway
            our_consistency.save(self.B_diag1, "psi_i_attmat")
            our_consistency.save(self.B_diag2, "psi_i_tokmat")
            our_consistency.save(valss, "psi_internal")

        return valss

    '''
    Compute embeddingss and embedding mask for all mentions
    INPUT:
    mentions: 1D python list of mentions
    n: len(mentions)
    RETURNS:
    3D (n,n_cands,d) tensor of embeddingss
    2D (n,n_cands) bool tensor mask
    '''

    def embeddings(self, mentions, n):
        embeddings = torch.zeros([n, SETTINGS.n_cands, SETTINGS.d]).to(
            SETTINGS.device)  # 3D (n,n_cands,d) tensor of embeddingss
        nan = 0  # use 0 as our nan-like value
        if SETTINGS.allow_nans:
            nan = float("nan")
        embeddings *= nan  # Actually a (n,n_cands,d) tensor of nans to begin with
        idss = torch.ones([n, SETTINGS.n_cands], dtype=torch.long).to(SETTINGS.device)  # 2D (n,n_cands) tensor of ids
        idss *= processeddata.unk_ent_id
        masks = torch.zeros([n, SETTINGS.n_cands], dtype=torch.bool).to(
            SETTINGS.device)  # 2D (n,n_cands) bool tensor of masks
        for m_idx, m in enumerate(mentions):
            if len(m.candidates) > 0:
                valss = torch.stack([e_i.ent_embedding_torch().to(SETTINGS.device) for e_i in m.candidates])
                # ids = torch.LongTensor([processeddata.ent_to_ent_id.get(e_i.text,processeddata.unk_ent_id) for e_i in m.candidates]).to(SETTINGS.device)
                embeddings[m_idx][0:len(valss)] = valss
                # idss[m_idx][0:len(valss)] = ids
                masks[m_idx][0:len(valss)] = torch.BoolTensor(
                    [processeddata.ent_to_ent_id.get(e_i.text, processeddata.unk_ent_id) != processeddata.unk_ent_id for
                     e_i in m.candidates]).to(SETTINGS.device)

        if SETTINGS.switches["consistency_psi"]:
            # our_consistency.save(idss,"psi_i_entid")
            our_consistency.save(masks.to(torch.float), "psi_i_entm")
        return embeddings, masks

    '''
    Compute embeddingss and embedding mask for all tokens in window around mentions
    INPUT:
    mentions: 1D python list of mentions
    n: len(mentions)
    RETURNS:
    3D (n,win,d) tensor of token embeddingss
    2D (n,win) bool tensor masks
    '''

    @staticmethod
    def token_embeddingss(mentions):
        l_contexts = [utils.string_to_token_embeddings(m.left_context, "left", SETTINGS.context_window_size)
                      for m in mentions]
        r_contexts = [utils.string_to_token_embeddings(m.right_context, "right", SETTINGS.context_window_size)
                      for m in mentions]
        unk_word_embedding = processeddata.word_id_to_embedding[processeddata.unk_word_id]
        contexts = [
            l_context + r_context
            if len(l_context) > 0 or len(r_context) > 0
            else [unk_word_embedding]
            for (l_context, r_context) in zip(l_contexts, r_contexts)
        ]

        # Pad all contexts on the right to equal size
        context_lens = [len(context) for context in contexts]
        max_len = max(context_lens)
        token_embeddingss = [context + [unk_word_embedding] * (max_len - len(context)) for context in contexts]
        token_maskss = [[1.] * context_len + [0.] * (max_len - context_len) for context_len in context_lens]

        token_embeddingss = torch.FloatTensor(token_embeddingss).to(SETTINGS.device)
        token_maskss = torch.BoolTensor(token_maskss).to(SETTINGS.device)

        if SETTINGS.switches["consistency_psi"]:
            l_contexts = [utils.string_to_token_ids(m.left_context, "left", SETTINGS.context_window_size) for m in
                          mentions]
            r_contexts = [utils.string_to_token_ids(m.right_context, "right", SETTINGS.context_window_size) for m in
                          mentions]
            contexts = [
                l_context + r_context
                if len(l_context) > 0 or len(r_context) > 0
                else [processeddata.unk_word_id]
                for (l_context, r_context) in zip(l_contexts, r_contexts)
            ]
            token_idss = [context + [processeddata.unk_word_id] * (max_len - len(context)) for context in contexts]
            our_consistency.save(torch.tensor(token_idss), "psi_i_tokid")
            our_consistency.save(token_maskss.to(torch.float), "psi_i_tokm")
        return token_embeddingss, token_maskss

    '''
    Calculate phi_k values for pairs of candidates in lists
    n: len(mentions)
    embeddingss: 3D (n,n_cands,d) tensor of embeddingss
    Returns: 
    5D (n_i,n_j,n_cands,n_cands,k) Tensor of phi_k values foreach k foreach candidate
    '''

    def phi_ksssss(self, n, embeddingss):
        # embeddingss is a 3D (n,n_cands,d) tensor
        # masks is a 2D (n,n_cands) bool tensor
        # R is a (k,d,d) tensor from (k,d) R_diag tensor
        valsss = embeddingss.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d])
        our_consistency.save(valsss, "phi_i_ent")
        # valss is a (n,1,n_cands,d) tensor
        # See image 2
        valsss = torch.matmul(valsss, self.R_diag.diag_embed())
        our_consistency.save(self.R_diag, "phi_i_rel")
        our_consistency.save(valsss, "phi_i_relent")
        # valss is (n,k,n_cands,d) tensor
        # Have (n,k,n_cands,d) and (n,n_cands,d) want (n,n,n_cands,n_cands,k)
        # (n,1,n_cands,d)*(n,1,n_cands,d,k) should do it?
        valsss = valsss.transpose(1, 2)
        # valsss is (n,n_cands,k,d)
        valsss = valsss.transpose(2, 3)
        # valsss is (n,n_cands,d,k)
        valsss = valsss.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d, SETTINGS.k])
        # valsss is (n,1,n_cands,d,k)
        embeddingss = embeddingss.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d])
        # embeddingss is (n,1,n_cands,d)
        # see image 1 (applied to 2D version, this was 3D)
        # valss = valss.matmul(embeddingss.T)
        valsss = torch.matmul(embeddingss, valsss)
        # valsss is (n,n,n_cands,n_cands,k)
        return valsss

    '''
    Calculates Phi for every candidate pair for every mention
    n: len(mentions)
    embeddingss: 3D (n,n_cands,d) tensor of embeddingss
    asss: 3D (n*n*k) matrix of a values 
    RETURN:
    4D (n_i,n_j,n_cands_i,n_cands_j) Tensor of phi values foreach candidate foreach i,j
    '''

    def phissss(self, n, embeddingss, ass):
        our_consistency.save("ment-norm", "mode")
        our_consistency.save("bilinear", "comp_mode")
        # 5D(n_i,n_j,n_cands_i,n_cands_j,k) , 4D (n_i,n_j,n_cands_i,n_cands_j)
        values = self.phi_ksssss(n, embeddingss)
        our_consistency.save(values, "phi_k")
        values *= ass.reshape([n, n, 1, 1, SETTINGS.k])  # broadcast along n_cands*n_cands
        values = smart_sum(values, dim=4)
        our_consistency.save(values, "phi")
        return values

    '''
    INPUT:
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    n: len(mentions)
    RETURNS:
    3D (n,n,k) tensor of a_ijk per each pair of (n) mentions
    '''

    def asss(self, fmcs, n):
        print("TEST1", fmcs.shape)
        x = self.exp_bracketssss(fmcs).clone()
        print("TEST2", x.shape)
        # x is (ni*nj*k)
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.REL_NORM:
            # X is (ni*nj*k)
            z_ijk = smart_sum(x, dim=2).reshape([n, n, 1])
            # Z_ijk is (ni*nj) sum, then a (ni*nj*1) broadcastable tensor
        else:
            # TODO - don't use their method (excluding j=i) it doesn't lead to the specified normalisation (summing to 1)
            # Instead just normalize by dividing by the sum (as expected, softmaxing)
            # Using their method we div by 0 if n=1
            # read brackets as (i,j',k)
            # Set diagonal to 0
            # brackets = x.clone()
            # eye = torch.eye(n,n)#n,n
            # antieye = 1-eye#for multiplying
            # antieye = antieye.reshape([n,n,1])#make 3d
            # brackets *= antieye
            z_ijk = smart_sum(x, dim=1).reshape([n, 1, SETTINGS.k])
            # Z_ijk is a (ni,k) sum, then a (ni*1*k) broadcastable tensor
        x /= z_ijk
        return x

    '''
    mentions:python list of all mentions
    RETURNS: 2D (n,d) tensor of f_m_c per mention'''

    def perform_fmcs(self, mentions):
        print("TESTX4", len(mentions))
        if SETTINGS.switches["snd_embs"]:
            print("TESTX5", len(mentions))
            half_window_size = SETTINGS.context_window_fmc // 2

            # Create id tensors for conll contexts
            print("TESTX6", len(mentions))
            print("TESTX7", type(mentions))
            mentions = mentions.copy()
            print("TESTX7", type(mentions))
            left_idss = [
                [
                    processeddata.word_to_word_id_snd.get(word, processeddata.unk_word_id_snd)
                    for word in m.conll_lctx
                ][-half_window_size:]
                for m in mentions
            ]
            print("TESTX8", len(mentions))
            print("TESTX9", type(mentions))
            mid_idss = [
                [
                    processeddata.word_to_word_id_snd.get(word, processeddata.unk_word_id_snd)
                    for word in m.conll_mctx
                ]
                for m in mentions
            ]
            right_idss = [
                list(reversed([
                                  processeddata.word_to_word_id_snd.get(word, processeddata.unk_word_id_snd)
                                  for word in m.conll_rctx
                              ][:half_window_size]))
                for m in mentions
            ]
            our_consistency.save(left_idss[0], "embs_i_lctx")
            # left_idss = [utils.stringToTokenIds(m.left_context, "left", window_size,special="snd") for m in
            #              mentions]
            lctx_ids_len = [len(leftIds) for leftIds in left_idss]
            max_lctx_len = max(lctx_ids_len)
            left_idss_ = [[processeddata.unk_word_id_snd] * (max_lctx_len - len(leftIds)) + leftIds for leftIds in
                         left_idss]
            our_consistency.save(torch.LongTensor(left_idss_), "fmc_i_lctx_ids")
            mctx_ids_len = [len(midIds) for midIds in mid_idss]
            max_mctx_len = max(mctx_ids_len)
            mid_idss_ = [midIds + [processeddata.unk_word_id_snd] * (max_mctx_len - len(midIds)) for midIds in mid_idss]
            our_consistency.save(torch.LongTensor(mid_idss_), "fmc_i_mctx_ids")
            rctx_ids_len = [len(rightIds) for rightIds in right_idss]
            max_rctx_len = max(rctx_ids_len)
            right_idss_ = [[processeddata.unk_word_id_snd] * (max_rctx_len - len(rightIds)) + rightIds for rightIds in
                          right_idss]
            our_consistency.save(torch.LongTensor(right_idss_), "fmc_i_rctx_ids")

            # embeddingss
            left_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                ]
                for ids in left_idss_
            ]
            mid_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                ]
                for ids in mid_idss_
            ]
            right_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                ]
                for ids in right_idss_
            ]
            our_consistency.save(torch.FloatTensor(left_embeddingss), "fmc_i_lctx_embs")
            our_consistency.save(torch.FloatTensor(mid_embeddingss), "fmc_i_mctx_embs")
            our_consistency.save(torch.FloatTensor(right_embeddingss), "fmc_i_rctx_embs")
            # readjust embeddingss to remove unknowns
            left_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                    if id != processeddata.unk_word_id_snd
                ]
                for ids in left_idss_
            ]
            mid_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                    if id != processeddata.unk_word_id_snd
                ]
                for ids in mid_idss_
            ]
            right_embeddingss = [
                [
                    processeddata.word_id_to_embedding_snd[id]
                    for id in ids
                    if id != processeddata.unk_word_id_snd
                ]
                for ids in right_idss_
            ]

            # arrays of summed embeddingss
            left_embedding_sums = []
            mid_embedding_sums = []
            right_embedding_sums = []
            for l, m, r in zip(left_embeddingss, mid_embeddingss, right_embeddingss):
                if len(l) == 0:  # just unknowns, so just make this 1 unknown exactly
                    l = processeddata.word_id_to_embedding_snd[processeddata.unk_word_id_snd].reshape(1, -1)
                if len(m) == 0:
                    m = processeddata.word_id_to_embedding_snd[processeddata.unk_word_id_snd].reshape(1, -1)
                if len(r) == 0:
                    r = processeddata.word_id_to_embedding_snd[processeddata.unk_word_id_snd].reshape(1, -1)
                # summation normalisation terms???
                left_size = len(l) + 1e-5
                mid_size = len(m) + 1e-5
                right_size = len(r) + 1e-5
                left_embedding_sums.append(torch.FloatTensor(l).sum(dim=0) / left_size)
                mid_embedding_sums.append(torch.FloatTensor(m).sum(dim=0) / mid_size)
                right_embedding_sums.append(torch.FloatTensor(r).sum(dim=0) / right_size)
            print("TESTA", len(mentions))
            print("TESTB", len(left_embedding_sums))
            # 2D n*d tensor of sum embedding for each mention
            left_tensor = torch.stack(left_embedding_sums).to(SETTINGS.device)
            mid_tensor = torch.stack(mid_embedding_sums).to(SETTINGS.device)
            right_tensor = torch.stack(right_embedding_sums).to(SETTINGS.device)
            our_consistency.save(left_tensor, "fmc_i_lctx_score")
            our_consistency.save(mid_tensor, "fmc_i_mctx_score")
            our_consistency.save(right_tensor, "fmc_i_rctx_score")
            #
            tensors = [left_tensor, mid_tensor, right_tensor]
        else:
            left_wordss = [m_i.left_context.split(" ") for m_i in mentions]
            mid_wordss = [m_i.text.split(" ") for m_i in mentions]
            right_wordss = [m_i.right_context.split(" ") for m_i in mentions]
            word_embedding_fn = lambda word: processeddata.word_id_to_embedding[
                processeddata.word_to_word_id.get(word,
                                                  processeddata.unk_word_id)]
            # 2D i*arbitrary python list of word embeddingss (each word embedding is numpy array)
            left_embeddingss = map_2D(word_embedding_fn, left_wordss)
            mid_embeddingss = map_2D(word_embedding_fn, mid_wordss)
            right_embeddingss = map_2D(word_embedding_fn, right_wordss)
            # 1D i python list of numpy arrays of summed embeddingss
            sum_fn = lambda embeddingsList: np.array(embeddingsList).sum(axis=0)
            left_embedding_sums = map_1D(sum_fn, left_embeddingss)
            mid_embedding_sums = map_1D(sum_fn, mid_embeddingss)
            right_embedding_sums = map_1D(sum_fn, right_embeddingss)
            # 2D i*d tensor of sum embedding for each mention
            left_tensor = torch.from_numpy(np.array(left_embedding_sums)).to(SETTINGS.device)
            mid_tensor = torch.from_numpy(np.array(mid_embedding_sums)).to(SETTINGS.device)
            right_tensor = torch.from_numpy(np.array(right_embedding_sums)).to(SETTINGS.device)
            #
            tensors = [left_tensor, mid_tensor, right_tensor]
        input_ = torch.cat(tensors, dim=1)
        input_ = input_.to(torch.float)  # make default tensor type for network
        torch.manual_seed(SETTINGS.SEED)
        our_consistency.save(input_, "bow_ctx_vecs")
        print("TESTC", input_.shape)
        torch.manual_seed(SETTINGS.SEED)
        our_consistency.save(list(self.f_m_c.parameters()), "fmc_preweight")
        our_consistency.save(self.f_m_c, "fmc_model")
        our_consistency.save(input_, "fmc_input")
        f = self.f_m_c(input_)
        print("TESTD", f.shape)
        our_consistency.save(f, "fmc_output")
        return f

    '''
    INPUT:
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    Returns:
    3D (n,n,k) tensor of 'exp brackets' values (equation 4 MRN paper)
    '''

    def exp_bracketssss(self, fmcs):
        # fmcs = torch.stack(fmcs)
        # each fmc is a 1D (d) tensor, fs is a 2D (n,d) tensor of fmc vals
        y = torch.matmul(fmcs, self.D_diag.diag_embed())  # mulmul not dot for non-1D dotting
        # y is a 3D (k,n,d) tensor
        y = y.transpose(0, 1)
        # y is a 3D (n,k,d) tensor)
        y = torch.matmul(y, fmcs.T).transpose(1, 2)
        # y is a 3D (n,n,k) tensor)
        our_consistency.save(y.permute(2, 0, 1), "exp_i_mentment")
        if SETTINGS.switches["exp_adjust"]:
            # for consistency checking, add identity * -1e10
            # NOTE - paper adds variable onto identity, does this actually make it change with gradient? or is gradient ignored as not param?
            # TODO - test how this works
            n = y.shape[0]
            eye = torch.eye(n).to(SETTINGS.device).view(n, n, 1)
            y_new = y.clone()
            y_new += eye * -1e10
            our_consistency.save(y_new.permute(2, 0, 1).clone(), "exp_i_mentment_1")
            y_new += eye * -1e10  # do it again!
            our_consistency.save(y_new.permute(2, 0, 1), "exp_i_mentment_2")
            x = y_new / np.math.sqrt(SETTINGS.d)
            our_consistency.save(x.permute(2, 0, 1), "exp_i_mentment_scaled")
            our_consistency.save(x.permute(2, 0, 1), "rel_ctx_ctx")  # TODO this will be different if >1000 mentions
            z = torch.exp(x)
            our_consistency.save(z.permute(2, 0, 1), "exp_i_mentment_probs")
        else:
            x = y / np.math.sqrt(SETTINGS.d)
            our_consistency.save(x.permute(2, 0, 1), "exp_i_mentment_scaled")
            our_consistency.save(x.permute(2, 0, 1), "rel_ctx_ctx")
            z = torch.exp(x)
            our_consistency.save(z.permute(2, 0, 1), "exp_i_mentment_probs")
        return z

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf

    '''
    Perform LBP iteration for all pairs of candidates
    m_bar: m_bar message values (n,n,n_cands) 3D tensor
    masks 2D (n,n_cands) boolean tensor mask
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,n_cands_i,n_cands_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,n_cands_j) of maximum message values for each i,j and each candidate for j
    '''

    def lbp_iteration_mvaluesss(self, m_bar, masks, n, lbp_inputs):
        # TODO - is the initial m_bar and the cancellation eye learned? MRN 235/238 mulrel_ranker.py
        # m_bar intuition: m_bar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # m_bar is a 3D (n_i,n_j,n_cands_j) tensor
        m_bar_sum = m_bar  # start by reading m_bar as k->i beliefs
        # (n_k,n_i,n_cands_i)
        # foreach j we will have a different sum, introduce j dimension
        m_bar_sum = m_bar_sum.repeat([n, 1, 1, 1])
        # (n_j,n_k,n_i,n_cands_i)
        # 0 out where j=k (do not want j->i(e'), set to 0 for all i,e')
        cancel = 1 - torch.eye(n, n).to(SETTINGS.device)  # (n_j,n_k) anti-identity
        cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
        m_bar_sum = m_bar_sum * cancel  # broadcast (n,n,1,1) to (n,n,n,n_cands), setting (n,n_cands) dim to 0 where j=k
        # (n_j,n_k,n_i,n_cands_i)
        # sum to a (n_i,n_j,n_cands_i) tensor of sums(over k) for each j,i,e'
        m_bar_sum = smart_sum(m_bar_sum, 1).transpose(0, 1)
        nan_test(m_bar_sum, "m_bar_sum")
        # lbp_inputs is phi+psi values, add to the m_bar sums to get the values in the max brackets
        #        lbp_inputs = lbp_inputs.permute(1,0,3,2)
        values = lbp_inputs + m_bar_sum.reshape(
            [n, n, SETTINGS.n_cands,
             1])  # broadcast (from (n_i,n_j,n_cands_i,1) to (n_i,n_j,n_cands_i,n_cands_j) tensor)
        minimum_value = values.min() - 1
        # Make brackets minimum value where e or e' don't exist
        values = values.clone()
        values[~masks.reshape([n, 1, SETTINGS.n_cands, 1]).repeat([1, n, 1, SETTINGS.n_cands])] = minimum_value
        values[~masks.reshape([n, 1, SETTINGS.n_cands, 1]).repeat([1, n, 1, SETTINGS.n_cands])] = minimum_value
        max_value = smart_max(values, dim=2)  # (n_i,n_j,n_cands_j) tensor of max values
        # max_value will be minimum_value if nonsensical, make it zero for now
        max_value[max_value == minimum_value] = 0
        # print("m_bar",max_value)
        return max_value

    '''
    Compute an iteration of LBP
    m_bar: m_bar message values (n,n,n_cands) 3D tensor
    masks 2D (n,n_cands) boolean tensor mask
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,n_cands_i,n_cands_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,n_cands_j) next m_bar
    '''

    def lbp_iteration_complete(self, m_bar, masks, n, lbp_inputs):
        # (n,n,n_cands_j) tensor
        m_valuessss = self.lbp_iteration_mvaluesss(m_bar, masks, n, lbp_inputs)
        nan_test(m_valuessss, "m_valuessss")
        # LBPEq11 - softmax m_valuessss
        # softmax invariant under translation, translate to around 0 to reduce float errors
        normalise_avg_to_zero_rows(m_valuessss, masks.reshape([1, n, SETTINGS.n_cands]), dim=2)
        exp_m_vals = m_valuessss.exp()
        exp_m_vals = exp_m_vals.clone()  # clone to prevent autograd errors (in-place modification next)
        set_mask_broadcastable(exp_m_vals, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)  # nans are 0
        softmax_denoms = smart_sum(exp_m_vals, dim=2)  # Eq 11 softmax denominator from LBP paper
        #        softmax_denoms[softmax_denoms == 0] = 1#divide by 1 not 0 if all values are 0 (e.g.
        #        print("DENOMS",softmax_denoms[:,30])
        #        print("EXPS",exp_m_vals[:,30,:])
        softmax_m_vals = exp_m_vals / softmax_denoms.reshape([n, n, 1])  # broadcast (n,n) to (n,n,n_cands)

        # Do Eq 11 (old mbars + m_valuessss to new mbars)
        damping_factor = SETTINGS.dropout_rate  # delta in the paper #TODO - I believe this is dropout_rate in the MRN code then?
        new_m_bar = m_bar.exp()
        new_m_bar = new_m_bar.mul(1 - damping_factor)  # dont use inplace after exp to prevent autograd error
        #        print("X1 ([0.25-]0.5)",new_m_bar)
        #        print("expm",exp_m_vals)
        other_bit = damping_factor * softmax_m_vals
        #        print("X2 (0-0.5)",other_bit)
        new_m_bar += other_bit
        #        print("X3 ([0.25-]0.5-1.0)",new_m_bar)
        set_mask_broadcastable(new_m_bar, ~masks.reshape([1, n, SETTINGS.n_cands]),
                               1)  # 'nan's become 0 after log (broadcast (1,n,n_cands) to (n,n,n_cands))
        new_m_bar = new_m_bar.log()
        #        print("X4 (-0.69 - 0)",new_m_bar)
        nan_test(new_m_bar, "new_m_bar")
        return new_m_bar

    def lbp_iteration_complete_backup(self, m_bar, masks, n, lbp_inputs):
        # (n,n,n_cands_j) tensor
        m_valuesss = self.lbp_iteration_mvaluesss(m_bar, masks, n, lbp_inputs)
        nan_test(m_valuesss, "m_valuesss")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avg_to_zero_rows(m_valuesss, masks.reshape([1, n, SETTINGS.n_cands]), dim=2)
        exp_m_vals = m_valuesss.exp()
        exp_m_vals = exp_m_vals.clone()  # clone to prevent autograd errors (in-place modification next)

        set_mask_broadcastable(exp_m_vals, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)  # nans are 0
        softmax_denoms = smart_sum(exp_m_vals, dim=2)  # Eq 11 softmax denominator from LBP paper

        # Do Eq 11 (old mbars + m_valuesss to new mbars)
        damping_factor = SETTINGS.dropout_rate  # delta in the paper #TODO - I believe this is dropout_rate in the MRN code then?
        new_m_bar = m_bar.exp()
        new_m_bar = new_m_bar.mul(1 - damping_factor)  # dont use inplace after exp to prevent autograd error
        #        print("X1 ([0.25-]0.5)",new_m_bar)
        #        print("expm",exp_m_vals)
        other_bit = damping_factor * (exp_m_vals / softmax_denoms.reshape([n, n, 1]))  # broadcast (n,n) to (n,n,n_cands)
        #        print("X2 (0-0.5)",other_bit)
        new_m_bar += other_bit
        #        print("X3 ([0.25-]0.5-1.0)",new_m_bar)
        set_mask_broadcastable(new_m_bar, ~masks.reshape([1, n, SETTINGS.n_cands]),
                               1)  # 'nan's become 0 after log (broadcast (1,n,n_cands) to (n,n,n_cands))
        new_m_bar = new_m_bar.log()
        #        print("X4 (-0.69 - 0)",new_m_bar)
        nan_test(new_m_bar, "new_m_bar")
        return new_m_bar

    '''
    Compute all LBP loops
    n: len(mentions)
    masks 2D (n,n_cands) boolean tensor mask
    f_m_cs: 2D (n,d) tensor of fmc values
    psiss: 2D tensor (n,n_cands) psi values per candidate per mention
    lbp_inputs: 4D tensor (n_i,n_j,n_cands_i,n_cands_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    2D (n,n_cands) tensor of ubar values
    '''

    def lbp_total(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*n_cands but unused cells will be 0/nan and ignored later
        debug("Computing initial m_bar for LBP")
        m_bar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        if SETTINGS.allow_nans:
            mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
            m_bar *= mbar_mask
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(m_bar, masks, n, lbp_inputs)
            m_bar = newmbar
        # Now compute u_bar
        nan_test(m_bar, "final m_bar")
        debug("Computing final u_bar out the back of LBP")
        anti_eye = 1 - torch.eye(n).to(SETTINGS.device)
        # read m_bar as (n_k,n_i,e_i)
        anti_eye = anti_eye.reshape([n, n, 1])  # reshape for broadcast
        m_bar = m_bar * anti_eye  # remove where k=i
        # make m_bar 0 where masked out
        set_mask_broadcastable(m_bar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        m_bar_sum = smart_sum(m_bar, 0)  # (n_i,e_i) sums
        u = psiss + m_bar_sum
        nan_test(u, "u")
        # mbarsum is sum of values between -inf,0
        # therefore mbarsum betweeen -inf,0
        # Note that psiss is unbounded

        # To compute softmax could use functional.softmax, however this cannot apply a mask.
        # Instead using softmax from difference from max (as large values produce inf, while small values produce 0 under exp)
        # Softmax is invariant under such a transformation (translation) - see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        u = u.clone()
        u[~masks] = u.clone().min()  # where masked out make min to ignore max
        u = u.clone()
        u -= u.clone().max(dim=1, keepdim=True)[0]
        u[~masks] = 0
        #        normalise_avgToZero_rowWise(u, masks, dim=1)
        nan_test(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0  # Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73, :])  # Mention 73, cand 3 has problems
            print(psiss[73, :])
            print(m_bar_sum[73, :])
            print(m_bar[:, 73, :])
            quit(0)
            # print((u>=88.72).nonzero())
        u[u >= 88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        u_bar = u.exp()  # 'nans' become 1
        if len(u_bar[u_bar == float("inf")]) > 0:
            print("u_bar has inf values after exp")
            print("Max value", u.max())
        u_bar = u_bar.clone()  # deepclone because performing in-place modification after exp
        u_bar[~masks] = 0  # reset nans to 0
        # Normalise u_bar (n,n_cands)
        nan_test(u_bar, "u_bar (in method)")
        u_bar_sum = smart_sum(u_bar, 1)  # (n_i) sums over candidates
        nan_test(u_bar, "u_bar (postsum)")
        nan_test(u_bar_sum, "u_bar_sum (postsum)")
        u_bar_sum = u_bar_sum.reshape([n, 1])  # (n_i,1) sum
        if len(u_bar_sum[u_bar_sum != u_bar_sum]) > 0:
            print("u_bar_sum has nan values")
        u_bar_sum_nans = u_bar_sum != u_bar_sum  # index tensor of where ubarsums is nan
        u_bar_sum[u_bar_sum_nans] = 1  # set to 1 to prevent division errors
        nan_test(u_bar_sum, "u_bar_sum")
        if (u_bar_sum < 0).sum() > 0:
            print("Found negative values in ubarusm", file=sys.stderr)
        u_bar_sum[u_bar_sum == 0] = 1  # Set 0 to 1 to prevent division error
        if len(u_bar_sum[u_bar_sum == float("inf")]) > 0:
            print("u_bar_sum has inf values")
        u_bar[u_bar_sum.repeat([1, SETTINGS.n_cands]) == float(
            "inf")] = 0  # Set to 0 when dividing by inf, repeat n_cands times across sum dim
        u_bar_sum[u_bar_sum == float("inf")] = 1  # Set to 1 to leave u_bar as 0 when dividing by inf
        if len(u_bar_sum[u_bar_sum < 1e-20]) > 0:
            print("u_bar has micro values")
            print(u_bar_sum)
            print(masks[3])
            print(u_bar[3])
            print(u_bar_sum[3])
            print(u[3])
            print(psiss[3])
            print(m_bar_sum[3])
            quit(0)
        u_bar /= u_bar_sum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            u_bar[~masks] = float("nan")
            u_bar[u_bar_sum_nans.reshape([n])] = float("nan")
        return u_bar

    def lbp_total_bk2(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*n_cands but unused cells will be 0/nan and ignored later
        debug("Computing initial m_bar for LBP")
        m_bar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        if SETTINGS.allow_nans:
            m_bar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
            nan = float("nan")
            m_bar_mask[m_bar_mask == 0] = nan  # make nan not 0
            m_bar *= m_bar_mask
        debug("Now doing LBP Loops")
        for loop_i in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loop_i + 1}/{SETTINGS.LBP_loops}")
            new_m_bar = self.lbp_iteration_complete(m_bar, masks, n, lbp_inputs)
            m_bar = new_m_bar
        # Now compute u_bar
        nan_test(m_bar, "final m_bar")
        debug("Computing final u_bar out the back of LBP")
        anti_eye = 1 - torch.eye(n).to(SETTINGS.device)
        # read m_bar as (n_k,n_i,e_i)
        anti_eye = anti_eye.reshape([n, n, 1])  # reshape for broadcast
        m_bar = m_bar * anti_eye  # remove where k=i
        # make m_bar 0 where masked out
        set_mask_broadcastable(m_bar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        m_bar_sum = smart_sum(m_bar, 0)  # (n_i,e_i) sums
        u = psiss + m_bar_sum
        nan_test(u, "u")
        # m_bar_sum is sum of values between -inf,0
        # therefore m_bar_sum betweeen -inf,0
        # Note that psiss is unbounded

        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avg_to_zero_rows(u, masks, dim=1)
        nan_test(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0  # Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73, :])  # Mention 73, cand 3 has problems
            print(psiss[73, :])
            print(m_bar_sum[73, :])
            print(m_bar[:, 73, :])
            quit(0)
            # print((u>=88.72).nonzero())
        u[u >= 88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        u_bar = u.exp()  # 'nans' become 1
        if len(u_bar[u_bar == float("inf")]) > 0:
            print("u_bar has inf values after exp")
            print("Max value", u.max())
        u_bar = u_bar.clone()  # deepclone because performing in-place modification after exp
        u_bar[~masks] = 0  # reset nans to 0
        # Normalise u_bar (n,n_cands)
        nan_test(u_bar, "u_bar (in method)")
        u_bar_sum = smart_sum(u_bar, 1)  # (n_i) sums over candidates
        nan_test(u_bar, "u_bar (postsum)")
        nan_test(u_bar_sum, "u_bar_sum (postsum)")
        u_bar_sum = u_bar_sum.reshape([n, 1])  # (n_i,1) sum
        if len(u_bar_sum[u_bar_sum != u_bar_sum]) > 0:
            print("u_bar_sum has nan values")
        u_bar_sum_nans = u_bar_sum != u_bar_sum  # index tensor of where ubarsums is nan
        u_bar_sum[u_bar_sum_nans] = 1  # set to 1 to prevent division errors
        nan_test(u_bar_sum, "u_bar_sum")
        if (u_bar_sum < 0).sum() > 0:
            print("Found negative values in ubarusm", file=sys.stderr)
        u_bar_sum[u_bar_sum == 0] = 1  # Set 0 to 1 to prevent division error
        if len(u_bar_sum[u_bar_sum == float("inf")]) > 0:
            print("u_bar_sum has inf values")
        u_bar[u_bar_sum.repeat([1, SETTINGS.n_cands]) == float(
            "inf")] = 0  # Set to 0 when dividing by inf, repeat n_cands times across sum dim
        u_bar_sum[u_bar_sum == float("inf")] = 1  # Set to 1 to leave u_bar as 0 when dividing by inf
        u_bar /= u_bar_sum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            u_bar[~masks] = float("nan")
            u_bar[u_bar_sum_nans.reshape([n])] = float("nan")
        return u_bar

    def lbp_total_bk(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*n_cands but unused cells will be 0/nan and ignored later
        debug("Computing initial m_bar for LBP")
        m_bar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        m_bar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            m_bar_mask[m_bar_mask == 0] = nan  # make nan not 0
        m_bar *= m_bar_mask
        debug("Now doing LBP Loops")
        for loop_i in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loop_i + 1}/{SETTINGS.LBP_loops}")
            new_m_bar = self.lbp_iteration_complete(m_bar, masks, n, lbp_inputs)
            m_bar = new_m_bar
        # Now compute u_bar
        nan_test(m_bar, "final m_bar")
        debug("Computing final u_bar out the back of LBP")
        anti_eye = 1 - torch.eye(n).to(SETTINGS.device)
        # read m_bar as (n_k,n_i,e_i)
        anti_eye = anti_eye.reshape([n, n, 1])  # reshape for broadcast
        m_bar = m_bar * anti_eye  # remove where k=i
        # make m_bar 0 where masked out
        set_mask_broadcastable(m_bar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        m_bar_sum = smart_sum(m_bar, 0)  # (n_i,e_i) sums
        u = psiss + m_bar_sum
        nan_test(u, "u")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avg_to_zero_rows(u, masks, dim=1)
        nan_test(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0  # Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73, :])  # Mention 73, cand 3 has problems
            print(psiss[73, :])
            print(m_bar_sum[73, :])
            print(m_bar[:, 73, :])
            quit(0)
            # print((u>=88.72).nonzero())
        u[u >= 88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        u_bar = u.exp()  # 'nans' become 1
        if len(u_bar[u_bar == float("inf")]) > 0:
            print("u_bar has inf values after exp")
            print("Max value", u.max())
        u_bar = u_bar.clone()  # deepclone because performing in-place modification after exp
        u_bar[~masks] = 0  # reset nans to 0
        # Normalise u_bar (n,n_cands)
        nan_test(u_bar, "u_bar (in method)")
        u_bar_sum = smart_sum(u_bar, 1)  # (n_i) sums over candidates
        nan_test(u_bar, "u_bar (postsum)")
        nan_test(u_bar_sum, "u_bar_sum (postsum)")
        u_bar_sum = u_bar_sum.reshape([n, 1])  # (n_i,1) sum
        if len(u_bar_sum[u_bar_sum != u_bar_sum]) > 0:
            print("u_bar_sum has nan values")
        u_bar_sum_nans = u_bar_sum != u_bar_sum  # index tensor of where ubarsums is nan
        u_bar_sum[u_bar_sum_nans] = 1  # set to 1 to prevent division errors
        nan_test(u_bar_sum, "u_bar_sum")
        if (u_bar_sum < 0).sum() > 0:
            print("Found negative values in ubarusm", file=sys.stderr)
        u_bar_sum[u_bar_sum == 0] = 1  # Set 0 to 1 to prevent division error
        if len(u_bar_sum[u_bar_sum == float("inf")]) > 0:
            print("u_bar_sum has inf values")
        u_bar[u_bar_sum.repeat([1, SETTINGS.n_cands]) == float(
            "inf")] = 0  # Set to 0 when dividing by inf, repeat n_cands times across sum dim
        u_bar_sum[u_bar_sum == float("inf")] = 1  # Set to 1 to leave u_bar as 0 when dividing by inf
        u_bar /= u_bar_sum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            u_bar[~masks] = float("nan")
            u_bar[u_bar_sum_nans.reshape([n])] = float("nan")
        return u_bar

    def forward(self, document: Document):
        mentions = document.mentions
        n = len(mentions)
        print("TESTX1", n)

        debug("Calculating embeddingss")
        embeddings, masks = self.embeddings(mentions, n)
        nan_test(embeddings, "embeddingss")
        mask_coverage = (masks.to(torch.float).sum() / (n * SETTINGS.n_cands)) * 100
        debug(f"Mask coverage {mask_coverage}%")
        print("TESTX2", len(mentions))

        debug("Calculating token embeddingss")
        token_embeddingss, token_maskss = self.token_embeddingss(mentions)
        nan_test(token_embeddingss, "token_embeddingss")

        debug("Calculating f_m_c values")
        print("TESTX3", len(mentions))
        f_m_cs = self.perform_fmcs(mentions)
        print("TEST0", f_m_cs.shape)
        nan_test(f_m_cs, "fmcs")

        debug("Calculating psi")
        psiss = self.psiss(n, embeddings, masks, token_embeddingss, token_maskss)  # 2d (n_i,n_cands_i) tensor
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.MENT_NORM and SETTINGS.switches["pad_enable"]:
            # Padding entity (they way they do)
            # add entity vec (randn(1)*0.1)
            #   self.pad_ent, self.pad_ctx

            pad_cand = Candidate(-1, 1, "#UNK#")  # Default p_e_m of 1, erroneous string and id
            pad_cand.ent_embedding_torch = lambda: self.pad_ent
            pad_unk = Candidate(processeddata.unk_ent_id, 0, "#UNK#")  # UNK

            cand_count = SETTINGS.n_cands_pem + SETTINGS.n_cands_ctx
            pad_cands = [pad_cand] + [pad_unk for _ in range(0, cand_count - 1)]
            pad_cand_embs = [c.ent_embedding_torch() for c in pad_cands]
            pad_cand_embs = torch.stack(pad_cand_embs).to(SETTINGS.device)
            pad_mask = [True] + [False for _ in range(0, cand_count - 1)]
            pad_psi = torch.FloatTensor([0 for _ in range(0, cand_count)]).to(SETTINGS.device)
            pad_mask = torch.BoolTensor(pad_mask).to(SETTINGS.device)

            pad_mention = Mention.from_data(-1, "", "", "", pad_cands, 0)
            document.mentions.append(pad_mention)  # document mentions not needed after this, do not pad

            embeddings = torch.cat([embeddings, pad_cand_embs.reshape(1, cand_count, 300)])
            # add mask (10000000)
            masks = torch.cat([masks, pad_mask.reshape(1, cand_count)])
            # add pem (10000000) [N/A]
            # add psi! 0000000
            psiss = torch.cat([psiss, pad_psi.reshape(1, cand_count)])
            # increment n obviously
            n += 1
            # [entity vec fmc other (randn(1)*0.1)]
            f_m_cs = torch.cat([f_m_cs, self.pad_ctx.reshape(1, 300)])
        our_consistency.save(f_m_cs, "ctx_vecs")

        debug("Calculating a values")
        asss = self.asss(f_m_cs, n)
        nan_test(asss, "asss")
        debug("Calculating phi")
        phis = self.phissss(n, embeddings, asss)  # 4d (n_i,n_j,n_cands_i,n_cands_j) tensor
        debug("Calculating lbp inputs")
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape(
            [n, 1, SETTINGS.n_cands, 1])  # broadcast (from (n_i,n_cands_i) to (n_i,n_j,n_cands_i,n_cands_j) tensor)
        nan_test(phis, "phis")
        nan_test(psiss, "psiss")
        nan_test(lbp_inputs, "lbp inputs")

        debug("Calculating u_bar(lbp) values")
        u_bar = self.lbp_total(n, masks, psiss, lbp_inputs)
        nan_test(u_bar, "u_bar")
        debug("Starting mention calculations")
        p_e_m = torch.zeros([n, SETTINGS.n_cands]).to(SETTINGS.device)
        for m_idx, m in enumerate(mentions):  # all mentions
            for e_idx, e in enumerate(m.candidates):  # candidate entities
                p_e_m[m_idx][e_idx] = e.initial_prob  # input from data
        # reshape to a (n*n_cands,2) tensor for use by the nn
        nan_test(p_e_m, "pem")
        u_bar = u_bar.reshape(n * SETTINGS.n_cands, 1)
        p_e_m = p_e_m.reshape(n * SETTINGS.n_cands, 1)
        inputs = torch.cat([u_bar, p_e_m], dim=1)
        inputs[~masks.reshape(n * SETTINGS.n_cands)] = 0
        nan_test(inputs, "g network inputs")
        p = self.g(inputs)
        if SETTINGS.allow_nans:
            p[~masks.reshape(n * SETTINGS.n_cands)] = float("nan")
        else:
            p[~masks.reshape(n * SETTINGS.n_cands)] = 0  # no chance
        p = p.reshape(n, SETTINGS.n_cands)  # back to original dims
        nan_test(p, "final p tensor")
        return p

# TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
