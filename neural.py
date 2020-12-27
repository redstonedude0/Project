# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import hyperparameters
import processeddata
from datastructures import Model, Mention, Document
from hyperparameters import SETTINGS
from utils import debug, map_2D, map_1D


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
        # TODO - set default parameter values properly
        self.register_parameter("B", torch.nn.Parameter(torch.diag(torch.ones(300))))
        self.register_parameter("R", torch.nn.Parameter(torch.stack(
            [torch.diag(torch.ones(300)), torch.diag(torch.ones(300)), torch.diag(torch.ones(300))])))  # todo k elem
        self.register_parameter("D", torch.nn.Parameter(
            torch.stack([torch.diag(torch.ones(300)), torch.diag(torch.ones(300)), torch.diag(torch.ones(300))])))

    # local and pairwise score functions (equation 3+section 3.1)
    '''
    Compute PSI for all candidates for a mention
    INPUT:
    m: the mention (provides candidates)
    f_m_c: 1D tensor (d) f_m_c score for the mention
    RETURN:
    1D tensor (arb_m) psi values per candidate for this mention
    '''

    def psis(self, m: Mention, f_m_c):
        c_i = m.left_context + m.right_context  # TODO - use context in f properly
        # I think f(c_i) is just the word embeddings in the 3 parts of context? needs to be a dim*1 vector?
        # f_c = self.f(c_i)#TODO - define f properly (Attention mechanism)
        f_c = f_m_c  # TODO - is f_m_c equal to f_c???
        embeddings = [e.entEmbeddingTorch() for e in m.candidates]
        embeddings = torch.stack(embeddings)
        vals = embeddings.matmul(self.B)
        vals = vals.matmul(f_c)
        return vals

    '''
    e_i:entity to calculate phi_k values for
    candidates:1D (arb) python list of entities to compute phi_k with
    Returns: 2D (arb,k) Tensor of phi_k values foreach k foreach candidate'''

    def phi_kss(self, e_i, candidates):
        vals = e_i.entEmbeddingTorch().T
        vals = vals.matmul(self.R).reshape(SETTINGS.k, SETTINGS.d)
        embeddings = torch.stack([e_j.entEmbeddingTorch() for e_j in candidates])
        vals = embeddings.matmul(vals.T)  # see image 1
        return vals

    '''
    Calculate phi_k values for pairs of candidates in lists
    candidates_i:1D (arb_i) python list of entities to compute phi_k with
    candidates_j:1D (arb_j) python list of entities to compute phi_k with
    Returns: 3D (arb_i,arb_j,k) Tensor of phi_k values foreach k foreach candidate'''

    def phi_ksss(self, candidates_i, candidates_j):
        valss = torch.stack([e_i.entEmbeddingTorch() for e_i in candidates_i])
        arb_i = len(candidates_i)
        # valss is a (arb_i,d) tensor
        valss = valss.matmul(self.R)
        # valss is (k,arb_i,d) tensor
        valss = valss.transpose(0, 1)
        # valss is (arb_i,k,d) tensor
        embeddings = torch.stack([e_j.entEmbeddingTorch() for e_j in candidates_j])
        valss = valss.matmul(embeddings.T)  # see image 1 (applied to 2D version, this is 3D)
        # valss is (arb_i,k,arb_j) tensor
        valss = valss.transpose(1, 2)
        # valss is (arb_i,arb_j,k) tensor
        return valss

    '''
    Calculates Phi for every e_j in a list
    e_i: specific entity to calculate it for
    candidates: 1D python list of candidates to compute phi value with
    m_i: mention for i
    m_j: mention for j
    ass: 3D (n*n*k) matrix of a values 
    RETURN: 1D (arb_j) Tensor of phi values foreach candidate for j
    '''

    def phis(self, e_i, candidates, i_idx, j_idx, ass):
        values = self.phi_kss(e_i, candidates)
        values *= ass[i_idx][j_idx]
        return values.sum(dim=1)

    '''
    Calculates Phi for every e_j and e_i in lists
    candidates_i: 1D python list of candidates to compute phi value with
    candidates_j: 1D python list of candidates to compute phi value with
    m_i: mention for i
    m_j: mention for j
    ass: 3D (n*n*k) matrix of a values 
    RETURN: 2D (arb_i,arb_j) Tensor of phi values foreach candidate for i,j
    '''

    def phiss(self, candidates_i, candidates_j, i_idx, j_idx, ass):
        # TODO can use float("NaN") to make it (7,7) not (arb,arb)
        values = self.phi_kss(e_i, candidates_j)
        values *= ass[i_idx][j_idx]
        return values.sum(dim=1)

    '''
    INPUT:
    mentions: python array of mentions
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    RETURNS:
    3D (n,n,k) tensor of a_ijk per each pair of (n) mentions
    '''

    def ass(self, mentions, fmcs):
        x = self.exp_bracketssss(mentions, fmcs)
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
        f = self.f_m_c(input_)
        print("OUTPUT:", f)
        return f

    '''
    INPUT:
    mentions: python list of all mentions
    fmcs: 2D (n,d) tensor of fmc values for each mention n
    Returns:
    3D (n,n,k) tensor of 'exp brackets' values (equation 4 MRN paper)
    '''

    def exp_bracketssss(self, mentions: List[Mention], fmcs):
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

    '''
    compute lbp values for 2 mentions and 1 candidate over all candidates for j
    mbar: mbar 3D tensor from t-1
    i,j: mentions (will calculate for all candidates of j)
    i_idx,j_idx: mbar indexes for i and j
    psis_j: 1D (arb_j) tensor of psi values per candidate for j
    arg:candidate for i
    RETURNS:
    maximum value (new message value) for the given i,j,arg(i candidate)
    '''

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf
    def lbp_iteration_individual(self, mbar, j, i_idx, j_idx, psis_j, ass, arg):
        # TODO ensure candidates is Gamma(i) - Gamma is the set of candidates for a mention?
        maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here?
        phis_ij = self.phis(arg, j.candidates, i_idx, j_idx, ass)  # TODO - check, shouldn't arg be for i, not j?
        values = psis_j
        values += phis_ij
        mbarmatrix = mbar  # Convert mbar to a torch tensor #TODO - keep it as a torch tensor - Do this, it will be easiest
        mbarslice = mbarmatrix[:, i_idx][:, 0:len(values)]  # Slice it to a k*7 tensor then down to k*arb
        mbarslice[j_idx] = 0  # 0 out side where k == j, so it can be summed inconsequentially
        mbarsums = mbarslice.sum(dim=0)
        values += mbarsums
        if len(values) != 0:  # Prevent identity error when tensor is empty
            maxValue = values.max()
        return maxValue

    '''
    Perform LBP iteration for set of candidates
    mbar: mbar message values (n,n,7) 3D tensor
    j,j_idx: j mention
    i_idx: i mention
    psis_j: 1D tensor (arb_j) psi values for j (per candidate)
    ass: 3D tensor (n,n,k) a values per i,j,k
    args: python list of candidates for i
    '''

    def lbp_iteration_individuals(self, mbar, j, i_idx, j_idx, psis_j, ass, args):
        # TODO ensure candidates is Gamma(i) - Gamma is the set of candidates for a mention?
        maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here?
        phis_ij = self.phiss(args, j.candidates, i_idx, j_idx, ass)  # TODO - check, shouldn't arg be for i, not j?
        values = psis_j
        values += phis_ij
        mbarmatrix = mbar  # Convert mbar to a torch tensor #TODO - keep it as a torch tensor - Do this, it will be easiest
        mbarslice = mbarmatrix[:, i_idx][:, 0:len(values)]  # Slice it to a k*7 tensor then down to k*arb
        mbarslice[j_idx] = 0  # 0 out side where k == j, so it can be summed inconsequentially
        mbarsums = mbarslice.sum(dim=0)
        values += mbarsums
        if len(values) != 0:  # Prevent identity error when tensor is empty
            maxValue = values.max()
        return maxValue

    '''
    INPUT:
    psis: python array of 1D tensors (i,arb) of psi values
    '''

    def lbp_iteration_complete(self, mbar, mentions, psis, ass):
        newmbar = mbar.clone()
        for i_idx, i in tqdm(enumerate(mentions)):
            for j_idx, j in enumerate(mentions):
                psis_j = psis[j_idx]
                mvalues = {}
                for arg in i.candidates:
                    newmval = self.lbp_iteration_individual(mbar, j, i_idx, j_idx, psis_j, ass, arg)
                    mvalues[arg.id] = newmval
                mvalsum = 0  # Eq 13 denominator from LBP paper
                for value in mvalues.values():
                    mvalsum += value.exp()
                for arg_idx, arg in enumerate(i.candidates):
                    # Bar (needs softmax)
                    dampingFactor = 0.5  # delta in the paper
                    mval = mvalues[arg.id]
                    bar = mbar[i_idx][j_idx][arg_idx].exp()
                    bar *= (1 - dampingFactor)
                    bar += dampingFactor * (mval.exp() / mvalsum)
                    bar = bar.log()
                    newmbar[i_idx][j_idx][arg_idx] = bar
        return newmbar

    def lbp_total(self, mentions: List[Mention], f_m_cs, ass):
        mbar = {}
        psis = []
        for (i, f_m_c) in zip(mentions, f_m_cs):
            psis_i = self.psis(i, f_m_c)
            psis.append(psis_i)
        # Note: Should be i*j*arb but arb dependent so i*j*7 but unused cells will be 0 and trimmed
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(len(mentions), len(mentions), 7)
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete(mbar, mentions, psis, ass)
            mbar = newmbar
        # Now compute ubar
        ubar = {}
        debug("Computing final ubar out the back of LBP")
        for (i_idx, (i, f_m_c)) in tqdm(enumerate(zip(mentions, f_m_cs))):
            ubar[i.id] = {}
            for arg_idx, arg in enumerate(i.candidates):
                sum = 0
                for k_idx, k in enumerate(mentions):
                    if k != i:
                        sum += mbar[k_idx][i_idx][arg_idx]
                u = psis[i_idx][arg_idx] + sum
                ubar[i.id][arg.id] = np.exp(u)
            sum = 0
            for arg in i.candidates:  # Gamma(mi)
                sum += ubar[i.id][arg.id]
            for arg in i.candidates:
                ubar[i.id][arg.id] /= sum  # Normalise
        return ubar

    def forward(self, document: Document):
        mentions = document.mentions
        debug("Calculating f_m_c values")
        f_m_cs = self.perform_fmcs(mentions)
        debug("Calculating a values")
        ass = self.ass(mentions, f_m_cs)
        debug("Calculating ubar(lbp) values")
        ubar = self.lbp_total(mentions, f_m_cs, ass)
        m: Mention
        p = {}
        debug("Starting mention loop")
        for m in mentions:  # all mentions
            p[m.id] = {}
            for e in m.candidates:  # candidate entities
                p_e_m = e.initial_prob  # input from data
                q_e_d = ubar[m.id][e.id]  # From LBP
                g = lambda x, y: None  # 2-layer NN #TODO
                p_e = g(q_e_d, p_e_m)
                p[m.id][e.id] = p_e
                #return all p_e for all m
        return p

#TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
