# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
from typing import List

import numpy as np
import torch
from torch import nn

import hyperparameters
import processeddata
from datastructures import Model, Mention, Document
from hyperparameters import SETTINGS
from utils import debug, map_2D, map_1D, maskedmax


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model'''


def train(model: Model):  # TODO - add params
    model.neuralNet: NeuralNet
    model.neuralNet.train()  # Set training flag
    print("Model output:")
    print("Selecting 1 document to train on")
    out = model.neuralNet(SETTINGS.dataset.documents[0])
    print(out)
    print("Done.")
    # TODO - train neural network using ADAM


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
        self.register_parameter("B", torch.nn.Parameter(torch.diag(torch.ones(300))))
        self.register_parameter("R", torch.nn.Parameter(torch.stack(
            [torch.diag(torch.ones(300)), torch.diag(torch.ones(300)), torch.diag(torch.ones(300))])))  # todo k elem
        self.register_parameter("D", torch.nn.Parameter(
            torch.stack([torch.diag(torch.ones(300)), torch.diag(torch.ones(300)), torch.diag(torch.ones(300))])))

    # local and pairwise score functions (equation 3+section 3.1)

    '''
    Compute PSI for all candidates for all mentions
    INPUT:
    n: len(mentions)
    embeddings: 3D (n,7,d) tensor of embeddings
    fmcs: 2D tensor(n,d) fmc values
    RETURN:
    2D tensor (n,7) psi values per candidate per mention
    #TODO - mask?
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
        embeddings = torch.zeros([n, 7, SETTINGS.d])  # 3D (n,7,d) tensor of embeddings
        masks = torch.zeros([n, 7], dtype=torch.bool)  # 2D (n,7) bool tensor of masks
        for m_idx, m in enumerate(mentions):
            valss = torch.stack([e_i.entEmbeddingTorch() for e_i in m.candidates])
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
        return values.sum(dim=4)

    '''
    INPUT:
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    RETURNS:
    3D (n,n,k) tensor of a_ijk per each pair of (n) mentions
    '''

    def ass(self, fmcs):
        x = self.exp_bracketssss(fmcs)
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            # X is (ni*nj*k)
            z_ijk = x.sum(dim=2)
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
        leftTensor = torch.from_numpy(np.array(leftEmbeddingSums))
        midTensor = torch.from_numpy(np.array(midEmbeddingSums))
        rightTensor = torch.from_numpy(np.array(rightEmbeddingSums))
        #
        tensors = [leftTensor, midTensor, rightTensor]
        input_ = torch.cat(tensors, dim=1).type(torch.Tensor)  # make default tensor type for network
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
    embeddings: 3D (n,7,d) tensor of embeddings
    masks: 2D (n,7) boolean tensor mask
    psiss: 2D tensor (n,7) all psi values for each mention i (per candidate e')
    ass: 3D tensor (n,n,k) a values per i,j,k
    RETURNS:
    3D tensor (n,n,7_j) of maximum message values for each i,j and each candidate for j
    '''

    def lbp_iteration_individualsss(self, mbar, n, embeddings, masks, psiss, ass):
        # mbar intuition: mbar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # TODO maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here (when i has no candidates)?
        phis = self.phissss(n, embeddings, ass)  # 4d (n_i,n_j,7_i,7_j) tensor
        values = phis  # values inside max{} brackets - Eq (10) LBP paper
        values = values + psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        # mbar is a 3D (n_i,n_j,7_j) tensor
        mbarsum = mbar  # start by reading mbar as k->i beliefs
        # (n_k,n_i,7_i)
        # foreach j we will have a different sum, introduce j dimension
        mbarsum = mbarsum.repeat([n, 1, 1, 1])
        # (n_j,n_k,n_i,7_i)
        # 0 out where j=k (do not want j->i(e'), set to 0 for all i,e')
        cancel = 1 - torch.eye(n, n)  # (n_j,n_k) anti-identity
        cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
        mbarsum = mbarsum * cancel  # broadcast (n,n,1,1) to (n,n,n,7), setting (n,7) dim to 0 where j=k
        # (n_j,n_k,n_i,7_i)
        # sum to a (n_j,n_i,7_i) tensor of sums(over k) for each j,i,e'
        mbarsum = mbarsum.sum(dim=1).transpose(0, 1)
        # (n_i,n_j,7_i)
        values = values + mbarsum.reshape([n, n, 7, 1])  # broadcast (from (n_i,n_j,7_i,1) to (n_i,n_j,7_i,7_j) tensor)
        #        if len(values) != 0:  # Prevent identity error when tensor is empty#TODO - how to prevent in multiple dimensions?
        # Masked max
        # reshape masks from (n_j,7_j) to (1,n_j,1,7_j) to broadcast to (n_i,n_j,7_i,7_j) tensor)
        maxValue = maskedmax(values, masks.reshape([1, n, 1, 7]), dim=2)[
            0]  # (n_i,n_j,7_j) tensor of max values ([0] gets max not argmax)
        return maxValue

    '''
    Compute an iteration of LBP
    mbar: mbar message values (n,n,7) 3D tensor
    n: len(mentions)
    embeddings: 3D (n,7,d) tensor of embeddings
    masks: 2D (n,7) boolean tensor mask
    psiss: 2D tensor (n,7) all psi values for each mention i (per candidate e')
    ass: 3D tensor (n,n,k) a values per i,j,k
    RETURNS:
    3D tensor (n,n,7_j) next mbar
    '''

    def lbp_iteration_complete(self, mbar, n, embeddings, masks, psiss, ass):
        # (n,n,7_j) tensor
        mvalues = self.lbp_iteration_individualsss(mbar, n, embeddings, masks, psiss, ass)
        expmvals = mvalues.exp()
        expmvals *= masks.type(torch.Tensor).reshape(
            [1, n, 7])  # 0 if masked out, reshape (1,n,7) to broadcast to (n_i,n_j,7_j)
        softmaxdenoms = expmvals.sum(dim=2)  # Eq 11 softmax denominator from LBP paper

        # Do Eq 11 (old mbars + mvalues to new mbars)
        dampingFactor = 0.5  # delta in the paper
        newmbar = mbar.exp()
        newmbar *= (1 - dampingFactor)
        otherbit = dampingFactor * (expmvals / softmaxdenoms.reshape([n, n, 1]))  # broadcast (n,n) to (n,n,7)
        newmbar += otherbit
        newmbar = newmbar.log()
        return newmbar

    '''
    Compute all LBP loops
    mentions: 1D (n) python list of mentions
    n: len(mentions)
    masks 2D (n,7) boolean tensor mask
    f_m_cs: 2D (n,d) tensor of fmc values
    ass: 3D (n,n,k) tensor of a values
    RETURNS:
    2D (n,7) tensor of ubar values
    '''

    def lbp_total(self, mentions: List[Mention], n, masks, embeddings, f_m_cs, ass):
        psiss = self.psiss(n, embeddings, f_m_cs)
        # Note: Should be i*j*arb but arb dependent so i*j*7 but unused cells will be 0 and trimmed
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(n, n, 7)
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(mbar, n, embeddings, masks, psiss, ass)
            mbar = newmbar
        # Now compute ubar
        debug("Computing final ubar out the back of LBP")
        antieye = 1 - torch.eye(n)
        # read mbar as (n_k,n_i,e_i)
        antieye = antieye.reshape([n, n, 1])  # reshape for broadcast
        mbar = mbar * antieye  # remove where k=i
        # make (n_i,7_i) mask of 1 where keep, 0 where delete
        mask = masks.type(torch.Tensor)
        mask = mask.reshape([1, n, 7])  # add dim1 for broadcasting
        mbar = mbar * mask
        mbar = mbar.sum(dim=0)  # (n_i,e_i) sums
        u = psiss + mbar
        ubar = u.exp()
        # Mask ubar (n,7)
        mask = mask.reshape([n, 7])
        ubar = ubar * mask
        # Normalise ubar (n,7)
        ubarsum = ubar.sum(dim=1)  # (n_i) sums over candidats
        ubarsum = ubarsum.reshape([n, 1])  # (n_i,1) sum
        ubar /= ubarsum  # broadcast (n_i,1) (n_i,7) to normalise
        return ubar

    def forward(self, document: Document):
        mentions = document.mentions
        n = len(mentions)
        debug("Calculating embeddings")
        embeddings, masks = self.embeddings(mentions, n)
        debug("Calculating f_m_c values")
        f_m_cs = self.perform_fmcs(mentions)
        debug("Calculating a values")
        ass = self.ass(f_m_cs)
        debug("Calculating ubar(lbp) values")
        ubar = self.lbp_total(mentions, n, masks, embeddings, f_m_cs, ass)
        debug("Starting mention calculations")
        p_e_m = torch.zeros([n, 7])
        for m_idx, m in enumerate(mentions):  # all mentions
            for e_idx, e in enumerate(m.candidates):  # candidate entities
                p_e_m[m_idx][e_idx] = e.initial_prob  # input from data
        # reshape to a (n*7,2) tensor for use by the nn
        ubar = ubar.reshape(n * 7, 1)
        p_e_m = p_e_m.reshape(n * 7, 1)
        p = self.g(torch.cat([ubar, p_e_m], dim=1))
        p.reshape(n, 7)  # back to original dims
        return p

#TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
