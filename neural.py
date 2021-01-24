# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import hyperparameters
import processeddata
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
    for doc_idx, document in enumerate(tqdm(SETTINGS.dataset.documents, unit="documents", file=sys.stdout)):
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
        loss += loss_regularisation(model.neuralNet.R, model.neuralNet.D)
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

    # Learn!
    print("Stepping...")
    #    optimizer.step()  # step - update parameters backwards as requried
    print("Done.")

    # Evaluate
    #    microPrecision = true_pos/(true_pos+false_pos)    #Equal to microF1 in this case
    eval = EvaluationMetrics()
    eval.loss = total_loss
    eval.accuracy = eval_correct / (eval_correct + eval_wrong)
    eval.accuracy_possible = possibleCorrect / total
    #    eval.correctRatio =
    #    eval.microF1 = microPrecision
    #    eval.correctRatio_possible = eval_correct / possibleCorrect
    #    eval.microF1_possible = true_pos/possibleCorrect
    return eval

def loss_document(document: Document, output):
    n = len(document.mentions)
    # MRN equation 7
    p_i_e = output  # 2D (n,7) tensor of p_i_e values
    truth_indices = [m.goldCandIndex() for m in document.mentions]
    # truth_indices is 1D (n) tensor of index of truth (0-6) (-1 for none)
    p_i_eSTAR = p_i_e.transpose(0, 1)[truth_indices].diagonal()
    # TODO - need to set piestar to 0 where truthindex -1?
    # broadcast (n,1) to (n,7)
    GammaEqn = SETTINGS.gamma - p_i_eSTAR.reshape([n, 1]) + p_i_e
    # Max each element with 0 (repeated to (n,7))
    h = torch.max(GammaEqn, torch.tensor(0.).to(SETTINGS.device).repeat([n, 7]))
    L = smartsum(h)  # ignore nans in loss function (outside of candidate range)
    return L


def loss_regularisation(R, D):
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
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )  # TODO dummy network - remove
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
        # TODO - set default parameter values properly
        self.register_parameter("B", torch.nn.Parameter(torch.diag(torch.ones(300).to(SETTINGS.device))))
        self.register_parameter("R", torch.nn.Parameter(torch.stack(
            [torch.diag(torch.ones(300).to(SETTINGS.device)), torch.diag(torch.ones(300).to(SETTINGS.device)), torch.diag(torch.ones(300).to(SETTINGS.device))])))  # todo k elem
        self.register_parameter("D", torch.nn.Parameter(
            torch.stack([torch.diag(torch.ones(300).to(SETTINGS.device)), torch.diag(torch.ones(300).to(SETTINGS.device)), torch.diag(torch.ones(300).to(SETTINGS.device))])))

    # local and pairwise score functions (equation 3+section 3.1)

    '''
    Compute PSI for all candidates for all mentions
    INPUT:
    n: len(mentions)
    embeddings: 3D (n,7,d) tensor of embeddings
    fmcs: 2D tensor(n,d) fmc values
    RETURN:
    2D tensor (n,7) psi values per candidate per mention
    '''

    def psiss(self, n, embeddings, fmcs):
        # c_i = m.left_context + m.right_context  # TODO - use context in f properly
        # I think f(c_i) is just the word embeddings in the 3 parts of context? needs to be a dim*1 vector?
        # f_c = self.f(c_i)#TODO - define f properly (Attention mechanism)
        fcs = fmcs  # TODO - is f_m_c equal to f_c???
        # embeddings is 3D (n,7,d) tensor
        # B is 2D (d,d) tensor
        vals = embeddings.matmul(self.B)
        # vals is 3D (n,7,d) tensor
        # fcs is 2D (n,d) tensor
        # (n,7,d)*(n,d,1) = (n,7,1)
        vals = vals.matmul(fcs.reshape([n, SETTINGS.d, 1])).reshape([n, 7])
        # vals is 2D (n,7) tensor
        return vals

    '''
    Compute embeddings and embedding mask for all mentions
    INPUT:
    mentions: 1D python list of mentions
    n: len(mentions)
    RETURNS:
    3D (n,7,d) tensor of embeddings
    2D (n,7) bool tensor mask
    '''

    def embeddings(self, mentions, n):
        embeddings = torch.zeros([n, 7, SETTINGS.d]).to(SETTINGS.device)  # 3D (n,7,d) tensor of embeddings
        nan = 0  # use 0 as our nan-like value
        if SETTINGS.allow_nans:
            nan = float("nan")
        embeddings *= nan  # Actually a (n,7,d) tensor of nans to begin with
        masks = torch.zeros([n, 7], dtype=torch.bool).to(SETTINGS.device)  # 2D (n,7) bool tensor of masks
        for m_idx, m in enumerate(mentions):
            if len(m.candidates) > 0:
                valss = torch.stack([e_i.entEmbeddingTorch().to(SETTINGS.device) for e_i in m.candidates])
                embeddings[m_idx][0:len(valss)] = valss
                masks[m_idx][0:len(valss)] = True
        return embeddings, masks

    '''
    Calculate phi_k values for pairs of candidates in lists
    n: len(mentions)
    embeddings: 3D (n,7,d) tensor of embeddings
    Returns: 
    5D (n_i,n_j,7,7,k) Tensor of phi_k values foreach k foreach candidate
    '''

    def phi_ksssss(self, n, embeddings):
        # embeddings is a 3D (n,7,d) tensor
        # masks is a 2D (n,7) bool tensor
        # R is a (k,d,d) tensor
        valsss = embeddings.reshape([n, 1, 7, SETTINGS.d])
        # valss is a (n,1,7,d) tensor
        # See image 2
        valsss = torch.matmul(valsss, self.R)
        # valss is (n,k,7,d) tensor
        # Have (n,k,7,d) and (n,7,d) want (n,n,7,7,k)
        # (n,1,7,d)*(n,1,7,d,k) should do it?
        valsss = valsss.transpose(1, 2)
        # valsss is (n,7,k,d)
        valsss = valsss.transpose(2, 3)
        # valsss is (n,7,d,k)
        valsss = valsss.reshape([n, 1, 7, SETTINGS.d, SETTINGS.k])
        # valsss is (n,1,7,d,k)
        embeddings = embeddings.reshape([n, 1, 7, SETTINGS.d])
        # embeddings is (n,1,7,d)
        # see image 1 (applied to 2D version, this was 3D)
        # valss = valss.matmul(embeddings.T)
        valsss = torch.matmul(embeddings, valsss)
        # valsss is (n,n,7,7,k)
        return valsss

    '''
    Calculates Phi for every candidate pair for every mention
    n: len(mentions)
    embeddings: 3D (n,7,d) tensor of embeddings
    ass: 3D (n*n*k) matrix of a values 
    RETURN:
    4D (n_i,n_j,7_i,7_j) Tensor of phi values foreach candidate foreach i,j
    '''

    def phissss(self, n, embeddings, ass):
        # 5D(n_i,n_j,7_i,7_j,k) , 4D (n_i,n_j,7_i,7_j)
        values = self.phi_ksssss(n, embeddings)
        values *= ass.reshape([n, n, 1, 1, SETTINGS.k])  # broadcast along 7*7
        values = smartsum(values, dim=4)
        return values

    '''
    INPUT:
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    RETURNS:
    3D (n,n,k) tensor of a_ijk per each pair of (n) mentions
    '''

    def ass(self, fmcs):
        x = self.exp_bracketssss(fmcs).clone()
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            # X is (ni*nj*k)
            z_ijk = smartsum(x, dim=2)
            # Z_ijk is (ni*nj) sum
            x_trans = x.transpose(0, 2).transpose(1, 2)
            # x_trans is (k*ni*nj)
            x_trans /= z_ijk
            x = x_trans.transpose(1, 2).transpose(0, 2)
            # x is (ni*nj*k)
        else:
            raise Exception("Unimplemented normalisation method")
            # TODO ment-norm Z
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
        y = torch.matmul(fmcs, self.D)  # mulmul not dot for non-1D dotting
        # y is a 3D (k,n,d) tensor
        y = y.transpose(0, 1)
        # y is a 3D (n,k,d) tensor)
        y = torch.matmul(y, fmcs.T).transpose(1, 2)
        # y is a 3D (n,n,k) tensor)
        x = y / np.math.sqrt(SETTINGS.d)
        return torch.exp(x)

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf

    '''
    Perform LBP iteration for all pairs of candidates
    mbar: mbar message values (n,n,7) 3D tensor
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,7_i,7_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,7_j) of maximum message values for each i,j and each candidate for j
    '''

    def lbp_iteration_mvaluesss(self, mbar, n, lbp_inputs):
        # TODO - is the initial mbar and the cancellation eye learned? MRN 235/238 mulrel_ranker.py
        # mbar intuition: mbar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # mbar is a 3D (n_i,n_j,7_j) tensor
        mbarsum = mbar  # start by reading mbar as k->i beliefs
        # (n_k,n_i,7_i)
        # foreach j we will have a different sum, introduce j dimension
        mbarsum = mbarsum.repeat([n, 1, 1, 1])
        # (n_j,n_k,n_i,7_i)
        # 0 out where j=k (do not want j->i(e'), set to 0 for all i,e')
        cancel = 1 - torch.eye(n, n).to(SETTINGS.device)  # (n_j,n_k) anti-identity
        cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
        mbarsum = mbarsum * cancel  # broadcast (n,n,1,1) to (n,n,n,7), setting (n,7) dim to 0 where j=k
        # (n_j,n_k,n_i,7_i)
        # sum to a (n_i,n_j,7_i) tensor of sums(over k) for each j,i,e'
        mbarsum = smartsum(mbarsum, 1).transpose(0, 1)
        nantest(mbarsum, "mbarsum")
        # lbp_inputs is phi+psi values, add to the mbar sums to get the values in the max brackets
        #        lbp_inputs = lbp_inputs.permute(1,0,3,2)
        values = lbp_inputs + mbarsum.reshape(
            [n, n, 7, 1])  # broadcast (from (n_i,n_j,7_i,1) to (n_i,n_j,7_i,7_j) tensor)
        maxValue = smartmax(values, dim=2)  # (n_i,n_j,7_j) tensor of max values
        #print("mbar",maxValue)
        return maxValue

    '''
    Compute an iteration of LBP
    mbar: mbar message values (n,n,7) 3D tensor
    masks 2D (n,7) boolean tensor mask
    n: len(mentions)
    lbp_inputs: 4D tensor (n_i,n_j,7_i,7_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    3D tensor (n,n,7_j) next mbar
    '''

    def lbp_iteration_complete(self, mbar, masks, n, lbp_inputs):
        # (n,n,7_j) tensor
        mvalues = self.lbp_iteration_mvaluesss(mbar, n, lbp_inputs)
        nantest(mvalues, "mvalues")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avgToZero_rowWise(mvalues, masks.reshape([1, n, 7]), dim=2)
        expmvals = mvalues.exp()
        expmvals = expmvals.clone()  # clone to prevent autograd errors (in-place modification next)

        setMaskBroadcastable(expmvals, ~masks.reshape([1, n, 7]), 0)  # nans are 0
        softmaxdenoms = smartsum(expmvals, dim=2)  # Eq 11 softmax denominator from LBP paper

        # Do Eq 11 (old mbars + mvalues to new mbars)
        dampingFactor = SETTINGS.dropout_rate  # delta in the paper #TODO - I believe this is dropout_rate in the MRN code then?
        newmbar = mbar.exp()
        newmbar = newmbar.mul(1 - dampingFactor)  # dont use inplace after exp to prevent autograd error
        #        print("X1 ([0.25-]0.5)",newmbar)
        #        print("expm",expmvals)
        otherbit = dampingFactor * (expmvals / softmaxdenoms.reshape([n, n, 1]))  # broadcast (n,n) to (n,n,7)
        #        print("X2 (0-0.5)",otherbit)
        newmbar += otherbit
        #        print("X3 ([0.25-]0.5-1.0)",newmbar)
        setMaskBroadcastable(newmbar, ~masks.reshape([1, n, 7]),
                             1)  # 'nan's become 0 after log (broadcast (1,n,7) to (n,n,7))
        newmbar = newmbar.log()
        #        print("X4 (-0.69 - 0)",newmbar)
        nantest(newmbar, "newmbar")
        return newmbar

    '''
    Compute all LBP loops
    n: len(mentions)
    masks 2D (n,7) boolean tensor mask
    f_m_cs: 2D (n,d) tensor of fmc values
    psiss: 2D tensor (n,7) psi values per candidate per mention
    lbp_inputs: 4D tensor (n_i,n_j,7_i,7_j) psi+phi values per i,j per candidates (e',e)
    RETURNS:
    2D (n,7) tensor of ubar values
    '''

    def lbp_total(self, n, masks, psiss, lbp_inputs):
        # Note: Should be i*j*arb but arb dependent so i*j*7 but unused cells will be 0/nan and ignored later
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(n, n, 7).to(SETTINGS.device)
        # should be nan if no candidate there (n_i,n_j,7_j)
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
        setMaskBroadcastable(mbar, ~masks.reshape([1, n, 7]), 0)
        mbar = smartsum(mbar, 0)  # (n_i,e_i) sums
        u = psiss + mbar
        nantest(u, "u")
        # softmax invariant under translation, translate to around 0 to reduce float errors
        # u+= 50
        # Normalise for each n across the row
        normalise_avgToZero_rowWise(u, masks, dim=1)
        nantest(u, "u (postNorm)")
        # TODO - what to translate by? why does 50 work here?
        ubar = u.exp()  # 'nans' become 1
        ubar = ubar.clone()  # deepclone because performing in-place modification after exp
        ubar[~masks] = 0  # reset nans to 0
        # Normalise ubar (n,7)
        nantest(ubar, "ubar (in method)")
        ubarsum = smartsum(ubar, 1)  # (n_i) sums over candidates
        ubarsum = ubarsum.reshape([n, 1])  # (n_i,1) sum
        ubarsumnans = ubarsum != ubarsum  # index tensor of where ubarsums is nan
        ubarsum[ubarsumnans] = 1  # set to 1 to prevent division errors
        nantest(ubarsum, "ubarsum")
        if (ubarsum<0).sum() > 0:
            print("Found negative values in ubarusm",file=sys.stderr)
        ubarsum[ubarsum==0]=1#Set 0 to 1 to prevent division error
        ubar /= ubarsum  # broadcast (n_i,1) (n_i,7) to normalise
        nantest(ubar, "ubar (post div)")
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
        maskCoverage = (masks.to(torch.float).sum() / (n * 7)) * 100
        debug(f"Mask coverage {maskCoverage}%")

        debug("Calculating f_m_c values")
        f_m_cs = self.perform_fmcs(mentions)
        nantest(f_m_cs, "fmcs")
        debug("Calculating a values")
        ass = self.ass(f_m_cs)
        nantest(ass, "ass")
        debug("Calculating lbp inputs")
        phis = self.phissss(n, embeddings, ass)  # 4d (n_i,n_j,7_i,7_j) tensor
        psiss = self.psiss(n, embeddings, f_m_cs)  # 2d (n_i,7_i) tensor
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        nantest(phis, "phis")
        nantest(psiss, "psiss")
        nantest(lbp_inputs, "lbp inputs")

        debug("Calculating ubar(lbp) values")
        ubar = self.lbp_total(n, masks, psiss, lbp_inputs)
        nantest(ubar, "ubar")
        debug("Starting mention calculations")
        p_e_m = torch.zeros([n, 7]).to(SETTINGS.device)
        for m_idx, m in enumerate(mentions):  # all mentions
            for e_idx, e in enumerate(m.candidates):  # candidate entities
                p_e_m[m_idx][e_idx] = e.initial_prob  # input from data
        # reshape to a (n*7,2) tensor for use by the nn
        nantest(p_e_m, "pem")
        ubar = ubar.reshape(n * 7, 1)
        p_e_m = p_e_m.reshape(n * 7, 1)
        inputs = torch.cat([ubar, p_e_m], dim=1)
        inputs[~masks.reshape(n * 7)] = 0
        nantest(inputs, "g network inputs")
        p = self.g(inputs)
        if SETTINGS.allow_nans:
            p[~masks.reshape(n * 7)] = float("nan")
        else:
            p[~masks.reshape(n * 7)] = 0  # no chance
        p = p.reshape(n, 7)  # back to original dims
        nantest(p,"final p tensor")
        return p

#TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
