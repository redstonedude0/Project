# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import hyperparameters
import processeddata
import utils
from datastructures import Model, Document, EvaluationMetrics
from hyperparameters import SETTINGS
from utils import debug, map_2D, map_1D, smartsum, smartmax, setMaskBroadcastable, \
    normalise_avgToZero_rowWise,nantest


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model, takes a learning rate'''


def train_OLD(model: Model, lr=SETTINGS.learning_rate_initial):
    # Prepare for training
    if SETTINGS.allow_nans:
        raise Exception("Fatal error - cannot learn with allow_nans enabled")
    model.neuralNet.train()  # Set training flag #TODO - why?
    torch.autograd.set_detect_anomaly(True)  # True for debugging to detect gradient anomolies

    # Initialise optimizer, calculate loss
    optimizer = torch.optim.Adam(model.neuralNet.parameters(), lr=lr)
    loss = loss_regularisation(model.neuralNet.R, model.neuralNet.D)
    eval_correct = 0
    eval_wrong = 0
    for doc_idx, document in enumerate(tqdm(SETTINGS.dataset.documents, unit="documents", file=sys.stdout)):
        out = model.neuralNet(document)
        if len(out[out != out]) > 1:
            # Dump
            print("Document index", doc_idx)
            print("Document id", document.id)
            print("Model output", out)
            raise Exception("Found nans in model output! Cannot proceed with learning")
        loss += loss_document(document, out)
        # Calculate evaluation metric data
        truth_indices = torch.tensor([m.goldCandIndex() for m in document.mentions]).to(SETTINGS.device)
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        best_cand_indices = out.max(dim=1)[1]  # index of maximum across candidates #1D(n) tensor
        same_list = truth_indices.eq(best_cand_indices)
        correct = same_list.sum().item()  # extract value from 0-dim tensor
        wrong = (~same_list).sum().item()
        eval_correct += correct
        eval_wrong += wrong

    # Learn!
    print("loss", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # step - update parameters backwards as requried
    print("Done.")

    # Evaluate
    eval = EvaluationMetrics()
    eval.loss = loss
    eval.correctRatio = eval_correct / (eval_correct + eval_wrong)
    return eval


'''
Attempted low memory version of train
'''


def train(model: Model, lr=SETTINGS.learning_rate_initial):
    # Prepare for training
    if SETTINGS.allow_nans:
        raise Exception("Fatal error - cannot learn with allow_nans enabled")
    model.neuralNet.to(SETTINGS.device)  # Move to correct device
    model.neuralNet.train()  # Set training flag #TODO - why?
    torch.autograd.set_detect_anomaly(True)  # True for debugging to detect gradient anomolies

    # Initialise optimizer, calculate loss
    optimizer = torch.optim.Adam(model.neuralNet.parameters(), lr=lr)
    optimizer.zero_grad()  # zero all gradients to clear buffers
    total_loss = 0
    # loss = loss_regularisation(model.neuralNet.R, model.neuralNet.D)
    # TODO - check - does the original paper add this regularisation once per loop?
    #    loss.backward()
    #total_loss += loss.item()
    eval_correct = 0
    eval_wrong = 0
    true_pos = 0  # guessed right valid index when valid index possible
    false_pos = 0  # guessed wrong index when valid index possible?
    false_neg = 0  # guessed wrong index when valid index possible?
    possibleCorrect = 0
    total = 0

    theirEval_truepos = 0
    theirEval_totalnonnilpredictions = 0
    theirEval_total = 0
    for doc_idx, document in enumerate(tqdm(SETTINGS.dataset_train.documents, unit="train_documents", file=sys.stdout)):
        if SETTINGS.lowmem:
            if len(document.mentions) > 200:
                print(
                    f"Unable to learn on document {doc_idx} ({len(document.mentions)} mentions would exceed memory limits)")
                continue
        try:
            out = model.neuralNet(document)
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
            continue  #next loop of document
        loss = loss_document(document, out)
        loss += loss_regularisation(model.neuralNet.R_diag, model.neuralNet.D_diag)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        # Calculate evaluation metric data
        truth_indices = torch.tensor([m.goldCandIndex() for m in document.mentions]).to(SETTINGS.device)
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
        possibleCorrect += possible.sum().item()
        total += len(same_list)

    if SETTINGS.loss_patched:
        eval_correct = 0
        eval_wrong = 0
        true_pos = 0
        false_pos = 0
        false_neg = 0
        possibleCorrect = 0
        total = 0
        #Do it their way
        for doc_idx, document in enumerate(tqdm(SETTINGS.dataset_eval.documents, unit="eval_documents", file=sys.stdout)):
            if SETTINGS.lowmem:
                if len(document.mentions) > 200:
                    print(
                        f"Unable to eval on document {doc_idx} ({len(document.mentions)} mentions would exceed memory limits)")
                    continue
            try:
                out = model.neuralNet(document)
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
            truth_indices = torch.tensor([m.goldCandIndex() for m in document.mentions]).to(SETTINGS.device)
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
            possibleCorrect += possible.sum().item()
            total += len(same_list)

            masks = [[True]*len(m.candidates)+[False]*(SETTINGS.n_cands-len(m.candidates)) for m in document.mentions]
            masks = torch.tensor(masks).to(SETTINGS.device)
            #Where the best candidate isn't real, try guessing 0
            best_cand_masks = masks.T[best_cand_indices].diagonal()
            best_cand_indices[~best_cand_masks] = 0
            #Where the best candidate still isn't real, guess -1
            best_cand_masks = masks.T[best_cand_indices].diagonal()
            best_cand_indices[~best_cand_masks] = -1
            same_list = truth_indices.eq(best_cand_indices)
            same_list[best_cand_indices==-1]=False#If Nil then not "the same"
            theirEval_truepos += same_list.sum().item()
            theirEval_totalnonnilpredictions += (best_cand_indices>=0).sum().item()
            theirEval_total += len(same_list)

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
        precision = theirEval_truepos / theirEval_totalnonnilpredictions
        recall = theirEval_truepos / theirEval_total
        theirAccuracy = 2*precision*recall/(precision+recall)
        print("precision",precision)
        print("recall",recall)
        print(eval.accuracy,"against",theirAccuracy)
        eval.accuracy = theirAccuracy
    eval.accuracy_possible = possibleCorrect / total
    #    eval.correctRatio =
    #    eval.microF1 = microPrecision
    #    eval.correctRatio_possible = eval_correct / possibleCorrect
    #    eval.microF1_possible = true_pos/possibleCorrect
    return eval

def loss_document(document: Document, output):
    if SETTINGS.loss_patched:
        truth_indices = torch.tensor([m.goldCandIndex() for m in document.mentions]).to(SETTINGS.device)
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        truth_indices[truth_indices==-1] = 0
        p_i_e = output  # 2D (n,n_cands) tensor of p_i_e values
        import torch.nn.functional as F
        loss = F.multi_margin_loss(p_i_e,truth_indices,margin=SETTINGS.gamma)
        return loss

    else:
        n = len(document.mentions)
        # MRN equation 7
        p_i_e = output  # 2D (n,n_cands) tensor of p_i_e values
        truth_indices = [m.goldCandIndex() for m in document.mentions]
        # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
        p_i_eSTAR = p_i_e.transpose(0, 1)[truth_indices].diagonal()
        # TODO - need to set piestar to 0 where truthindex -1?
        # broadcast (n,1) to (n,n_cands)
        GammaEqn = SETTINGS.gamma - p_i_eSTAR.reshape([n, 1]) + p_i_e
        # Max each element with 0 (repeated to (n,n_cands))
        h = torch.max(GammaEqn, torch.tensor(0.).to(SETTINGS.device).repeat([n, SETTINGS.n_cands]))
        L = smartsum(h)  # ignore nans in loss function (outside of candidate range)
        return L


def loss_regularisation(R_diag, D_diag):
    R = R_diag.diag_embed()
    D = D_diag.diag_embed()
    Regularisation_R = 0
    for i in range(0, SETTINGS.k):
        for j in range(0, i):
            # Do for every ordered pair i,j
            # If the paper means to do for all possible combinations i,j ignoring order then change this
            # TODO
            Regularisation_R += dist(R[i], R[j])
    Regularisation_D = 0
    for i in range(0, SETTINGS.k):
        for j in range(0, i):
            # TODO Change if necesssary
            Regularisation_D += dist(D[i], D[j])
    Regularisation_R *= SETTINGS.lambda1
    Regularisation_D *= SETTINGS.lambda2
    Regularisation = Regularisation_R + Regularisation_D
    return Regularisation


def dist(x, y):
    # p-norm with p=2 is the Euclidean/Frobenius norm given by .norm()
    xpart = x / x.norm()
    ypart = y / y.norm()
    return (xpart - ypart).norm()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        torch.manual_seed(0)
        self.f = nn.Sequential(
            nn.MultiheadAttention(100, 100, SETTINGS.dropout_rate)
            # TODO - absolutely no idea how to do this, this is a filler for now
        )  # Attention mechanism to achieve feature representation
        self.f_m_c = nn.Sequential(
            nn.Linear(900, 300).float(),  # TODO what dimensions?
            nn.Tanh(),
            nn.Dropout(p=SETTINGS.dropout_rate),
        ).float()
        g_hid_dims = 100  # TODO - this is the default used by the paper's code, not mentioned anywhere in paper
        self.g = nn.Sequential(
            nn.Linear(2, g_hid_dims),
            nn.ReLU(),
            nn.Linear(g_hid_dims, 1)
            # TODO - this is the 2-layer nn 'g' referred to in the paper, called score_combine in mulrel_ranker.py
        )
        B_diag = torch.ones(300).to(SETTINGS.device)
        B_diag1 = torch.ones(300).to(SETTINGS.device)
        B_diag2 = torch.ones(300).to(SETTINGS.device)
        #randn~Norm(0,1)
        R_diag = torch.randn(SETTINGS.k,300).to(SETTINGS.device)*0.1# todo k elem
        D_diag = torch.randn(SETTINGS.k,300).to(SETTINGS.device)*0.1# todo also k elems

        if not SETTINGS.rel_specialinit:
            R_diag += 1

        self.register_parameter("B_diag", torch.nn.Parameter(B_diag))
        self.register_parameter("B_diag1", torch.nn.Parameter(B_diag1))
        self.register_parameter("B_diag2", torch.nn.Parameter(B_diag2))
        self.register_parameter("R_diag", torch.nn.Parameter(R_diag))
        self.register_parameter("D_diag", torch.nn.Parameter(D_diag))
        #TODO - check the distribution of D, paper seems to be different than impl? (0.01ment/0.1rel)
        #TODO - check for B, paper seems to have 2 matrices to do ctx score? (ctxattrranker.forward)

    # local and pairwise score functions (equation 3+section 3.1)

    '''
    Compute CTX embedding f(c)
    INPUT:
    n: len(mentions)
    embeddingss: 3D (n,n_cands,d) tensor of embeddings of entities
    tokenEmbeddingss: 3D tensor(n,win,d) tensor of embeddings of tokens in window around each mention
    tokenMaskss: 3D bool tensor(n,win) tensor of masks
    RETURN:
    3D tensor (n,n_cands,d) of context embeddings
    '''
    def f_c(self,n,embeddingss,tokenEmbeddingss,tokenMaskss):
        window_size = tokenMaskss.shape[1]#Window size is dynamic, up to 100
        #For each mention i, we compute the correlation for each token t, for each candidate (of i) c
        #embeddingss (n,n_cands,d) * diag weighting.diag() (d,d) * tokenEmbeddingss (n,win,d)
        weightedTokenEmbeddingss_ = torch.matmul(self.B_diag2.diag_embed(),tokenEmbeddingss.transpose(1,2)) # (n,d,win)
        tokenEntityScoresss = torch.matmul(embeddingss,weightedTokenEmbeddingss_) # (n,n_cands,win)
        #set to -1e10 for unknown tokens
        tokenEntityScoresss[~tokenMaskss.reshape([n,1,window_size]).repeat([1,SETTINGS.n_cands,1])] = -1e10

        #Let the score of each token be the best score across all candidates (ignore indexes)
        tokenScoress,_ = torch.max(tokenEntityScoresss,dim=1) #(n,win)

        #Take the top (default 25) scores (unsorted!), with corresponding id
        best_scoress,best_scoress_idx = torch.topk(tokenScoress, dim=1, k=min(SETTINGS.attention_token_count, window_size)) # (n,25), (n,25) [technically <25 if small window, assuming 25 for sake of annotations]

        #Take the actual embeddings for the top 25 tokens
        best_tokenss = torch.gather(tokenEmbeddingss, dim=1,
                                         index=best_scoress_idx.view(n,-1,1).repeat(1, 1, SETTINGS.d)) # (n,25,d)

        #Scale each tokens embedding by prob
        token_probss = nn.functional.softmax(best_scoress,dim=1) #(n,25)
        token_probss = token_probss / torch.sum(token_probss,dim=1,keepdim=True) #(n,25), linear normalise because they do


        best_tokenss = best_tokenss * token_probss.view(n,-1,1)#multiplication will broadcast, (n,25,d)

        #Sum the 25-best window to achieve a context embedding weighted by token probability
        context_embeddings = torch.sum(best_tokenss,dim=1) #(n,d)

        return context_embeddings

    '''
    Compute PSI for all candidates for all mentions
    INPUT:
    n: len(mentions)
    embeddings: 3D (n,n_cands,d) tensor of embeddings
    tokenEmbeddingss: 3D tensor(n,win,d) tensor of embeddings of tokens in window around each mention
    tokenMaskss: 3D bool tensor(n,win) tensor of masks
    RETURN:
    2D tensor (n,n_cands) psi values per candidate per mention
    '''

    def psiss(self, n, embeddings, tokenEmbeddings,tokenMaskss):
        #Compute context embeddings f(c)
        fcs = self.f_c(n,embeddings,tokenEmbeddings,tokenMaskss) #(n,d)
        # embeddings is 3D (n,n_cands,d) tensor
        # B is 2D (d,d) tensor (from (d) B_diag tensor)

        #embeddingss (n,n_cands,d) * diag weighting.diag() (d,d) * fcs (n,d)
        weighted_context_embeddings = torch.matmul(self.B_diag1.diag_embed(),fcs.T).T # (n,d)
        weighted_context_embeddings = weighted_context_embeddings.view(n,SETTINGS.d,1) #(n,d,1)
        valss = torch.matmul(embeddings,weighted_context_embeddings) #(n,n_cands,1)

        #remove extra dim
        valss = valss.view(n,SETTINGS.n_cands)
        # vals2 is 2D (n,n_cands) tensor
        return valss

    '''
    Compute embeddings and embedding mask for all mentions
    INPUT:
    mentions: 1D python list of mentions
    n: len(mentions)
    RETURNS:
    3D (n,n_cands,d) tensor of embeddings
    2D (n,n_cands) bool tensor mask
    '''

    def embeddings(self, mentions, n):
        embeddings = torch.zeros([n, SETTINGS.n_cands, SETTINGS.d]).to(SETTINGS.device)  # 3D (n,n_cands,d) tensor of embeddings
        nan = 0  # use 0 as our nan-like value
        if SETTINGS.allow_nans:
            nan = float("nan")
        embeddings *= nan  # Actually a (n,n_cands,d) tensor of nans to begin with
        masks = torch.zeros([n, SETTINGS.n_cands], dtype=torch.bool).to(SETTINGS.device)  # 2D (n,n_cands) bool tensor of masks
        for m_idx, m in enumerate(mentions):
            if len(m.candidates) > 0:
                valss = torch.stack([e_i.entEmbeddingTorch().to(SETTINGS.device) for e_i in m.candidates])
                embeddings[m_idx][0:len(valss)] = valss
                masks[m_idx][0:len(valss)] = True
        return embeddings, masks

    '''
    Compute embeddings and embedding mask for all tokens in window around mentions
    INPUT:
    mentions: 1D python list of mentions
    n: len(mentions)
    RETURNS:
    3D (n,win,d) tensor of token embeddings
    2D (n,win) bool tensor masks
    '''

    def tokenEmbeddingss(self, mentions):
        contextss = [[m.left_context,m.right_context] for m in mentions]
        wordsss = [[ctx.strip().split() for ctx in contexts] for contexts in contextss]
        #TODO normalise word
        wordsss = [[[processeddata.wordid2embedding[processeddata.word2wordid[word]] for word in words if utils.is_important_dict_word(word)]for words in wordss]for wordss in wordsss]
        l_contexts = [wordss[0][-(SETTINGS.context_window_size // 2):] for wordss in wordsss]
        r_contexts = [wordss[1][:(SETTINGS.context_window_size // 2)] for wordss in wordsss]
        unkwordembedding = processeddata.wordid2embedding[processeddata.unkwordid]
        contexts = [
            l_context+r_context
            if len(l_context) > 0 or len(r_context) > 0
            else [unkwordembedding]
            for (l_context,r_context) in zip(l_contexts,r_contexts)
        ]

        #Pad all contexts on the right to equal size
        context_lens = [len(context) for context in contexts]
        max_len = max(context_lens)
        tokenEmbeddingss = [context + [unkwordembedding] * (max_len - len(context)) for context in contexts]
        tokenMaskss = [[1.] * context_len + [0.] * (max_len - context_len) for context_len in context_lens]

        tokenEmbeddingss = torch.FloatTensor(tokenEmbeddingss).to(SETTINGS.device)
        tokenMaskss = torch.BoolTensor(tokenMaskss).to(SETTINGS.device)
        return tokenEmbeddingss, tokenMaskss

    '''
    Calculate phi_k values for pairs of candidates in lists
    n: len(mentions)
    embeddings: 3D (n,n_cands,d) tensor of embeddings
    Returns: 
    5D (n_i,n_j,n_cands,n_cands,k) Tensor of phi_k values foreach k foreach candidate
    '''

    def phi_ksssss(self, n, embeddings):
        # embeddings is a 3D (n,n_cands,d) tensor
        # masks is a 2D (n,n_cands) bool tensor
        # R is a (k,d,d) tensor from (k,d) R_diag tensor
        valsss = embeddings.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d])
        # valss is a (n,1,n_cands,d) tensor
        # See image 2
        valsss = torch.matmul(valsss, self.R_diag.diag_embed())
        # valss is (n,k,n_cands,d) tensor
        # Have (n,k,n_cands,d) and (n,n_cands,d) want (n,n,n_cands,n_cands,k)
        # (n,1,n_cands,d)*(n,1,n_cands,d,k) should do it?
        valsss = valsss.transpose(1, 2)
        # valsss is (n,n_cands,k,d)
        valsss = valsss.transpose(2, 3)
        # valsss is (n,n_cands,d,k)
        valsss = valsss.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d, SETTINGS.k])
        # valsss is (n,1,n_cands,d,k)
        embeddings = embeddings.reshape([n, 1, SETTINGS.n_cands, SETTINGS.d])
        # embeddings is (n,1,n_cands,d)
        # see image 1 (applied to 2D version, this was 3D)
        # valss = valss.matmul(embeddings.T)
        valsss = torch.matmul(embeddings, valsss)
        # valsss is (n,n,n_cands,n_cands,k)
        return valsss

    '''
    Calculates Phi for every candidate pair for every mention
    n: len(mentions)
    embeddings: 3D (n,n_cands,d) tensor of embeddings
    ass: 3D (n*n*k) matrix of a values 
    RETURN:
    4D (n_i,n_j,n_cands_i,n_cands_j) Tensor of phi values foreach candidate foreach i,j
    '''

    def phissss(self, n, embeddings, ass):
        # 5D(n_i,n_j,n_cands_i,n_cands_j,k) , 4D (n_i,n_j,n_cands_i,n_cands_j)
        values = self.phi_ksssss(n, embeddings)
        values *= ass.reshape([n, n, 1, 1, SETTINGS.k])  # broadcast along n_cands*n_cands
        values = smartsum(values, dim=4)
        return values

    '''
    INPUT:
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    n: len(mentions)
    RETURNS:
    3D (n,n,k) tensor of a_ijk per each pair of (n) mentions
    '''

    def ass(self, fmcs, n):
        x = self.exp_bracketssss(fmcs).clone()
        # x is (ni*nj*k)
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            # X is (ni*nj*k)
            z_ijk = smartsum(x, dim=2).reshape([n, n, 1])
            # Z_ijk is (ni*nj) sum, then a (ni*nj*1) broadcastable tensor
        else:
            #TODO - don't use their method (excluding j=i) it doesn't lead to the specified normalisation (summing to 1)
            #Instead just normalize by dividing by the sum (as expected, softmaxing)
            #Using their method we div by 0 if n=1
            #read brackets as (i,j',k)
            #Set diagonal to 0
            #brackets = x.clone()
            #eye = torch.eye(n,n)#n,n
            #antieye = 1-eye#for multiplying
            #antieye = antieye.reshape([n,n,1])#make 3d
            #brackets *= antieye
            z_ijk = smartsum(x,dim=1).reshape([n,1,SETTINGS.k])
            #Z_ijk is a (ni,k) sum, then a (ni*1*k) broadcastable tensor
        x /= z_ijk
        return x

    '''
    mentions:python list of all mentions
    RETURNS: 2D (n,d) tensor of f_m_c per mention'''

    def perform_fmcs(self, mentions):
        leftWordss = [m_i.left_context.split(" ") for m_i in mentions]
        midWordss = [m_i.text.split(" ") for m_i in mentions]
        rightWordss = [m_i.right_context.split(" ") for m_i in mentions]
        wordEmbedding = lambda word: processeddata.wordid2embedding[
            processeddata.word2wordid.get(word,
                                          processeddata.unkwordid)]
        # 2D i*arbitrary python list of word embeddings (each word embedding is numpy array)
        leftEmbeddingss = map_2D(wordEmbedding, leftWordss)
        midEmbeddingss = map_2D(wordEmbedding, midWordss)
        rightEmbeddingss = map_2D(wordEmbedding, rightWordss)
        # 1D i python list of numpy arrays of summed embeddings
        sumFunc = lambda embeddingsList: np.array(embeddingsList).sum(axis=0)
        leftEmbeddingSums = map_1D(sumFunc, leftEmbeddingss)
        midEmbeddingSums = map_1D(sumFunc, midEmbeddingss)
        rightEmbeddingSums = map_1D(sumFunc, rightEmbeddingss)
        # 2D i*d tensor of sum embedding for each mention
        leftTensor = torch.from_numpy(np.array(leftEmbeddingSums)).to(SETTINGS.device)
        midTensor = torch.from_numpy(np.array(midEmbeddingSums)).to(SETTINGS.device)
        rightTensor = torch.from_numpy(np.array(rightEmbeddingSums)).to(SETTINGS.device)
        #
        tensors = [leftTensor, midTensor, rightTensor]
        input_ = torch.cat(tensors, dim=1)
        input_ = input_.to(torch.float)  # make default tensor type for network
        torch.manual_seed(0)
        f = self.f_m_c(input_)
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
        x = y / np.math.sqrt(SETTINGS.d)
#        print("MIN",self.D.min())
#        print("MAX",self.D.max())
#        #problem value is x[3,3,0]
#        if len(x) > 4:
#            print("LX VALS",x[3,3,:])
        return torch.exp(x)

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf

    '''
    Perform LBP iteration for all pairs of candidates
    mbar: mbar message values (n,n,n_cands) 3D tensor
    masks 2D (n,n_cands) boolean tensor mask
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,n_cands_i,n_cands_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,n_cands_j) of maximum message values for each i,j and each candidate for j
    '''

    def lbp_iteration_mvaluesss(self, mbar,masks, n, lbp_inputs):
        # TODO - is the initial mbar and the cancellation eye learned? MRN 235/238 mulrel_ranker.py
        # mbar intuition: mbar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # mbar is a 3D (n_i,n_j,n_cands_j) tensor
        mbarsum = mbar  # start by reading mbar as k->i beliefs
        # (n_k,n_i,n_cands_i)
        # foreach j we will have a different sum, introduce j dimension
        mbarsum = mbarsum.repeat([n, 1, 1, 1])
        # (n_j,n_k,n_i,n_cands_i)
        # 0 out where j=k (do not want j->i(e'), set to 0 for all i,e')
        cancel = 1 - torch.eye(n, n).to(SETTINGS.device)  # (n_j,n_k) anti-identity
        cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
        mbarsum = mbarsum * cancel  # broadcast (n,n,1,1) to (n,n,n,n_cands), setting (n,n_cands) dim to 0 where j=k
        # (n_j,n_k,n_i,n_cands_i)
        # sum to a (n_i,n_j,n_cands_i) tensor of sums(over k) for each j,i,e'
        mbarsum = smartsum(mbarsum, 1).transpose(0, 1)
        nantest(mbarsum, "mbarsum")
        # lbp_inputs is phi+psi values, add to the mbar sums to get the values in the max brackets
        #        lbp_inputs = lbp_inputs.permute(1,0,3,2)
        values = lbp_inputs + mbarsum.reshape(
            [n, n, SETTINGS.n_cands, 1])  # broadcast (from (n_i,n_j,n_cands_i,1) to (n_i,n_j,n_cands_i,n_cands_j) tensor)
        minimumValue = values.min()-1
        #Make brackets minimum value where e or e' don't exist
        values = values.clone()
        values[~masks.reshape([n,1,SETTINGS.n_cands,1]).repeat([1,n,1,SETTINGS.n_cands])] = minimumValue
        values[~masks.reshape([n,1,SETTINGS.n_cands,1]).repeat([1,n,1,SETTINGS.n_cands])] = minimumValue
        maxValue = smartmax(values, dim=2)  # (n_i,n_j,n_cands_j) tensor of max values
        #maxValue will be minimumValue if nonsensical, make it zero for now
        maxValue[maxValue==minimumValue] = 0
        #print("mbar",maxValue)
        return maxValue

    '''
    Compute an iteration of LBP
    mbar: mbar message values (n,n,n_cands) 3D tensor
    masks 2D (n,n_cands) boolean tensor mask
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,n_cands_i,n_cands_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,n_cands_j) next mbar
    '''

    def lbp_iteration_complete(self, mbar, masks, n, lbp_inputs):
        # (n,n,n_cands_j) tensor
        mvalues = self.lbp_iteration_mvaluesss(mbar,masks, n, lbp_inputs)
        nantest(mvalues, "mvalues")
        #LBPEq11 - softmax mvalues
        # softmax invariant under translation, translate to around 0 to reduce float errors
        normalise_avgToZero_rowWise(mvalues, masks.reshape([1, n, SETTINGS.n_cands]), dim=2)
        expmvals = mvalues.exp()
        expmvals = expmvals.clone()  # clone to prevent autograd errors (in-place modification next)
        setMaskBroadcastable(expmvals, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)  # nans are 0
        softmaxdenoms = smartsum(expmvals, dim=2)  # Eq 11 softmax denominator from LBP paper
        softmaxmvalues = expmvals/softmaxdenoms.reshape([n,n,1])# broadcast (n,n) to (n,n,n_cands)

        # Do Eq 11 (old mbars + mvalues to new mbars)
        dampingFactor = SETTINGS.dropout_rate  # delta in the paper #TODO - I believe this is dropout_rate in the MRN code then?
        newmbar = mbar.exp()
        newmbar = newmbar.mul(1 - dampingFactor)  # dont use inplace after exp to prevent autograd error
        #        print("X1 ([0.25-]0.5)",newmbar)
        #        print("expm",expmvals)
        otherbit = dampingFactor * softmaxmvalues
        #        print("X2 (0-0.5)",otherbit)
        newmbar += otherbit
        #        print("X3 ([0.25-]0.5-1.0)",newmbar)
        setMaskBroadcastable(newmbar, ~masks.reshape([1, n, SETTINGS.n_cands]),
                             1)  # 'nan's become 0 after log (broadcast (1,n,n_cands) to (n,n,n_cands))
        newmbar = newmbar.log()
        #        print("X4 (-0.69 - 0)",newmbar)
        nantest(newmbar, "newmbar")
        return newmbar

    def lbp_iteration_complete_bk(self, mbar, masks, n, lbp_inputs):
        # (n,n,n_cands_j) tensor
        mvalues = self.lbp_iteration_mvaluesss(mbar,masks, n, lbp_inputs)
        nantest(mvalues, "mvalues")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avgToZero_rowWise(mvalues, masks.reshape([1, n, SETTINGS.n_cands]), dim=2)
        expmvals = mvalues.exp()
        expmvals = expmvals.clone()  # clone to prevent autograd errors (in-place modification next)

        setMaskBroadcastable(expmvals, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)  # nans are 0
        softmaxdenoms = smartsum(expmvals, dim=2)  # Eq 11 softmax denominator from LBP paper

        # Do Eq 11 (old mbars + mvalues to new mbars)
        dampingFactor = SETTINGS.dropout_rate  # delta in the paper #TODO - I believe this is dropout_rate in the MRN code then?
        newmbar = mbar.exp()
        newmbar = newmbar.mul(1 - dampingFactor)  # dont use inplace after exp to prevent autograd error
        #        print("X1 ([0.25-]0.5)",newmbar)
        #        print("expm",expmvals)
        otherbit = dampingFactor * (expmvals / softmaxdenoms.reshape([n, n, 1]))  # broadcast (n,n) to (n,n,n_cands)
        #        print("X2 (0-0.5)",otherbit)
        newmbar += otherbit
        #        print("X3 ([0.25-]0.5-1.0)",newmbar)
        setMaskBroadcastable(newmbar, ~masks.reshape([1, n, SETTINGS.n_cands]),
                             1)  # 'nan's become 0 after log (broadcast (1,n,n_cands) to (n,n,n_cands))
        newmbar = newmbar.log()
        #        print("X4 (-0.69 - 0)",newmbar)
        nantest(newmbar, "newmbar")
        return newmbar

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
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        if SETTINGS.allow_nans:
            mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
            mbar *= mbar_mask
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(mbar, masks, n, lbp_inputs)
            mbar = newmbar
        # Now compute ubar
        nantest(mbar, "final mbar")
        debug("Computing final ubar out the back of LBP")
        antieye = 1 - torch.eye(n).to(SETTINGS.device)
        # read mbar as (n_k,n_i,e_i)
        antieye = antieye.reshape([n, n, 1])  # reshape for broadcast
        mbar = mbar * antieye  # remove where k=i
        # make mbar 0 where masked out
        setMaskBroadcastable(mbar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        mbarsum = smartsum(mbar, 0)  # (n_i,e_i) sums
        u = psiss + mbarsum
        nantest(u, "u")
        #mbarsum is sum of values between -inf,0
        #therefore mbarsum betweeen -inf,0
        #Note that psiss is unbounded


        #To compute softmax could use functional.softmax, however this cannot apply a mask.
        #Instead using softmax from difference from max (as large values produce inf, while small values produce 0 under exp)
        #Softmax is invariant under such a transformation (translation) - see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        u = u.clone()
        u[~masks]=u.clone().min() #where masked out make min to ignore max
        u = u.clone()
        u -= u.clone().max(dim=1,keepdim=True)[0]
        u[~masks]=0
#        normalise_avgToZero_rowWise(u, masks, dim=1)
        nantest(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0#Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73,:])#Mention 73, cand 3 has problems
            print(psiss[73,:])
            print(mbarsum[73,:])
            print(mbar[:,73,:])
            quit(0)
            #print((u>=88.72).nonzero())
        u[u>=88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        ubar = u.exp()  # 'nans' become 1
        if len(ubar[ubar == float("inf")]) > 0:
            print("ubar has inf values after exp")
            print("Max value",u.max())
        ubar = ubar.clone()  # deepclone because performing in-place modification after exp
        ubar[~masks] = 0  # reset nans to 0
        # Normalise ubar (n,n_cands)
        nantest(ubar, "ubar (in method)")
        ubarsum = smartsum(ubar, 1)  # (n_i) sums over candidates
        nantest(ubar, "ubar (postsum)")
        nantest(ubarsum, "ubarsum (postsum)")
        ubarsum = ubarsum.reshape([n, 1])  # (n_i,1) sum
        if len(ubarsum[ubarsum !=ubarsum]) > 0:
            print("ubarsum has nan values")
        ubarsumnans = ubarsum != ubarsum  # index tensor of where ubarsums is nan
        ubarsum[ubarsumnans] = 1  # set to 1 to prevent division errors
        nantest(ubarsum, "ubarsum")
        if (ubarsum<0).sum() > 0:
            print("Found negative values in ubarusm",file=sys.stderr)
        ubarsum[ubarsum==0]=1#Set 0 to 1 to prevent division error
        if len(ubarsum[ubarsum == float("inf")]) > 0:
            print("ubarsum has inf values")
        ubar[ubarsum.repeat([1,SETTINGS.n_cands])==float("inf")] = 0#Set to 0 when dividing by inf, repeat n_cands times across sum dim
        ubarsum[ubarsum==float("inf")] = 1#Set to 1 to leave ubar as 0 when dividing by inf
        if len(ubarsum[ubarsum<1e-20]) > 0:
            print("ubar has micro values")
            print(ubarsum)
            print(masks[3])
            print(ubar[3])
            print(ubarsum[3])
            print(u[3])
            print(psiss[3])
            print(mbarsum[3])
            quit(0)
        ubar /= ubarsum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            ubar[~masks] = float("nan")
            ubar[ubarsumnans.reshape([n])] = float("nan")
        return ubar


    def lbp_total_bk2(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*n_cands but unused cells will be 0/nan and ignored later
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        if SETTINGS.allow_nans:
            mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
            mbar *= mbar_mask
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(mbar, masks, n, lbp_inputs)
            mbar = newmbar
        # Now compute ubar
        nantest(mbar, "final mbar")
        debug("Computing final ubar out the back of LBP")
        antieye = 1 - torch.eye(n).to(SETTINGS.device)
        # read mbar as (n_k,n_i,e_i)
        antieye = antieye.reshape([n, n, 1])  # reshape for broadcast
        mbar = mbar * antieye  # remove where k=i
        # make mbar 0 where masked out
        setMaskBroadcastable(mbar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        mbarsum = smartsum(mbar, 0)  # (n_i,e_i) sums
        u = psiss + mbarsum
        nantest(u, "u")
        #mbarsum is sum of values between -inf,0
        #therefore mbarsum betweeen -inf,0
        #Note that psiss is unbounded

        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avgToZero_rowWise(u, masks, dim=1)
        nantest(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0#Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73,:])#Mention 73, cand 3 has problems
            print(psiss[73,:])
            print(mbarsum[73,:])
            print(mbar[:,73,:])
            quit(0)
            #print((u>=88.72).nonzero())
        u[u>=88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        ubar = u.exp()  # 'nans' become 1
        if len(ubar[ubar == float("inf")]) > 0:
            print("ubar has inf values after exp")
            print("Max value",u.max())
        ubar = ubar.clone()  # deepclone because performing in-place modification after exp
        ubar[~masks] = 0  # reset nans to 0
        # Normalise ubar (n,n_cands)
        nantest(ubar, "ubar (in method)")
        ubarsum = smartsum(ubar, 1)  # (n_i) sums over candidates
        nantest(ubar, "ubar (postsum)")
        nantest(ubarsum, "ubarsum (postsum)")
        ubarsum = ubarsum.reshape([n, 1])  # (n_i,1) sum
        if len(ubarsum[ubarsum !=ubarsum]) > 0:
            print("ubarsum has nan values")
        ubarsumnans = ubarsum != ubarsum  # index tensor of where ubarsums is nan
        ubarsum[ubarsumnans] = 1  # set to 1 to prevent division errors
        nantest(ubarsum, "ubarsum")
        if (ubarsum<0).sum() > 0:
            print("Found negative values in ubarusm",file=sys.stderr)
        ubarsum[ubarsum==0]=1#Set 0 to 1 to prevent division error
        if len(ubarsum[ubarsum == float("inf")]) > 0:
            print("ubarsum has inf values")
        ubar[ubarsum.repeat([1,SETTINGS.n_cands])==float("inf")] = 0#Set to 0 when dividing by inf, repeat n_cands times across sum dim
        ubarsum[ubarsum==float("inf")] = 1#Set to 1 to leave ubar as 0 when dividing by inf
        ubar /= ubarsum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            ubar[~masks] = float("nan")
            ubar[ubarsumnans.reshape([n])] = float("nan")
        return ubar

    def lbp_total_bk(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*n_cands but unused cells will be 0/nan and ignored later
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(n, n, SETTINGS.n_cands).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,n_cands_j)
        mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
        mbar *= mbar_mask
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            debug(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(mbar, masks, n, lbp_inputs)
            mbar = newmbar
        # Now compute ubar
        nantest(mbar, "final mbar")
        debug("Computing final ubar out the back of LBP")
        antieye = 1 - torch.eye(n).to(SETTINGS.device)
        # read mbar as (n_k,n_i,e_i)
        antieye = antieye.reshape([n, n, 1])  # reshape for broadcast
        mbar = mbar * antieye  # remove where k=i
        # make mbar 0 where masked out
        setMaskBroadcastable(mbar, ~masks.reshape([1, n, SETTINGS.n_cands]), 0)
        mbarsum = smartsum(mbar, 0)  # (n_i,e_i) sums
        u = psiss + mbarsum
        nantest(u, "u")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avgToZero_rowWise(u, masks, dim=1)
        nantest(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        if len(u[u == float("inf")]) > 0:
            print("u has inf values before exp")
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp1")
        u[~masks] = 0#Incase these values get in the way
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp2")
            print(u[73,:])#Mention 73, cand 3 has problems
            print(psiss[73,:])
            print(mbarsum[73,:])
            print(mbar[:,73,:])
            quit(0)
            #print((u>=88.72).nonzero())
        u[u>=88.72] = 0
        if len(u[u >= 88.72]) > 0:
            print("u has values that will become inf after exp3")
        ubar = u.exp()  # 'nans' become 1
        if len(ubar[ubar == float("inf")]) > 0:
            print("ubar has inf values after exp")
            print("Max value",u.max())
        ubar = ubar.clone()  # deepclone because performing in-place modification after exp
        ubar[~masks] = 0  # reset nans to 0
        # Normalise ubar (n,n_cands)
        nantest(ubar, "ubar (in method)")
        ubarsum = smartsum(ubar, 1)  # (n_i) sums over candidates
        nantest(ubar, "ubar (postsum)")
        nantest(ubarsum, "ubarsum (postsum)")
        ubarsum = ubarsum.reshape([n, 1])  # (n_i,1) sum
        if len(ubarsum[ubarsum !=ubarsum]) > 0:
            print("ubarsum has nan values")
        ubarsumnans = ubarsum != ubarsum  # index tensor of where ubarsums is nan
        ubarsum[ubarsumnans] = 1  # set to 1 to prevent division errors
        nantest(ubarsum, "ubarsum")
        if (ubarsum<0).sum() > 0:
            print("Found negative values in ubarusm",file=sys.stderr)
        ubarsum[ubarsum==0]=1#Set 0 to 1 to prevent division error
        if len(ubarsum[ubarsum == float("inf")]) > 0:
            print("ubarsum has inf values")
        ubar[ubarsum.repeat([1,SETTINGS.n_cands])==float("inf")] = 0#Set to 0 when dividing by inf, repeat n_cands times across sum dim
        ubarsum[ubarsum==float("inf")] = 1#Set to 1 to leave ubar as 0 when dividing by inf
        ubar /= ubarsum  # broadcast (n_i,1) (n_i,n_cands) to normalise
        if SETTINGS.allow_nans:
            ubar[~masks] = float("nan")
            ubar[ubarsumnans.reshape([n])] = float("nan")
        return ubar

    def forward(self, document: Document):
        mentions = document.mentions
        n = len(mentions)

        debug("Calculating embeddings")
        embeddings, masks = self.embeddings(mentions, n)
        nantest(embeddings, "embeddings")
        maskCoverage = (masks.to(torch.float).sum() / (n * SETTINGS.n_cands)) * 100
        debug(f"Mask coverage {maskCoverage}%")

        debug("Calculating token embeddings")
        tokenEmbeddingss, tokenMaskss = self.tokenEmbeddingss(mentions)
        nantest(tokenEmbeddingss, "tokenEmbeddingss")

        debug("Calculating f_m_c values")
        f_m_cs = self.perform_fmcs(mentions)
        nantest(f_m_cs, "fmcs")
        debug("Calculating a values")
        ass = self.ass(f_m_cs,n)
        nantest(ass, "ass")
        debug("Calculating lbp inputs")
        phis = self.phissss(n, embeddings, ass)  # 4d (n_i,n_j,n_cands_i,n_cands_j) tensor
        psiss = self.psiss(n, embeddings, tokenEmbeddingss,tokenMaskss)  # 2d (n_i,n_cands_i) tensor
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, SETTINGS.n_cands, 1])  # broadcast (from (n_i,n_cands_i) to (n_i,n_j,n_cands_i,n_cands_j) tensor)
        nantest(phis, "phis")
        nantest(psiss, "psiss")
        nantest(lbp_inputs, "lbp inputs")

        debug("Calculating ubar(lbp) values")
        ubar = self.lbp_total(n, masks, psiss, lbp_inputs)
        nantest(ubar, "ubar")
        debug("Starting mention calculations")
        p_e_m = torch.zeros([n, SETTINGS.n_cands]).to(SETTINGS.device)
        for m_idx, m in enumerate(mentions):  # all mentions
            for e_idx, e in enumerate(m.candidates):  # candidate entities
                p_e_m[m_idx][e_idx] = e.initial_prob  # input from data
        # reshape to a (n*n_cands,2) tensor for use by the nn
        nantest(p_e_m, "pem")
        ubar = ubar.reshape(n * SETTINGS.n_cands, 1)
        p_e_m = p_e_m.reshape(n * SETTINGS.n_cands, 1)
        inputs = torch.cat([ubar, p_e_m], dim=1)
        inputs[~masks.reshape(n * SETTINGS.n_cands)] = 0
        nantest(inputs, "g network inputs")
        p = self.g(inputs)
        if SETTINGS.allow_nans:
            p[~masks.reshape(n * SETTINGS.n_cands)] = float("nan")
        else:
            p[~masks.reshape(n * SETTINGS.n_cands)] = 0  # no chance
        p = p.reshape(n, SETTINGS.n_cands)  # back to original dims
        nantest(p,"final p tensor")
        return p

#TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
