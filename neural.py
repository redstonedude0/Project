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
    1D tensor (arb) psi values per candidate for this mention
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

    '''e_i,e_j:entities (embedding can be obtained)
    Returns: 1D Tensor of phi_k values foreach k'''
    def phi_ks(self, e_i, e_j):
        vals = e_i.entEmbeddingTorch().T
        vals = vals.matmul(self.R).reshape(SETTINGS.k, SETTINGS.d)
        vals = vals.matmul(e_j.entEmbeddingTorch())
        return vals

    def phi(self, e_i, e_j, m_i, m_j):
        values = self.phi_ks(e_i, e_j)
        values *= self.a(m_i, m_j)
        return values.sum()

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
    Calculates Phi for every e_j in a list
    e_i: specific entity to calculate it for
    candidates: 1D python list of candidates to compute phi value with
    m_i: mention for i
    m_j: mention for j
    RETURN: 1D () Tensor of phi values foreach candidate j
    '''

    def phis(self, e_i, candidates, m_i, m_j):
        values = self.phi_kss(e_i, candidates)
        values *= self.a(m_i, m_j)
        return values.sum(dim=1)

    def a(self, m_i, m_j):
        x = self.exp_bracketss(m_i, m_j)
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            z_ijk = x.sum()
        else:
            raise Exception("Unimplemented normalisation method")
            # TODO ment-norm Z
        return x / z_ijk

    def perform_fmc(self, m_i):
        leftWords = m_i.left_context.split(" ")
        midWords = m_i.text.split(" ")
        rightWords = m_i.right_context.split(" ")
        wordEmbedding = lambda word: processeddata.wordid2embedding[
            processeddata.word2wordid.get(word,
                                          processeddata.unkwordid)]
        leftEmbeddings = list(map(wordEmbedding, leftWords))
        midEmbeddings = list(map(wordEmbedding, midWords))
        rightEmbeddings = list(map(wordEmbedding, rightWords))
        leftTensors = torch.from_numpy(np.array(leftEmbeddings))
        midTensors = torch.from_numpy(np.array(midEmbeddings))
        rightTensors = torch.from_numpy(np.array(rightEmbeddings))
        leftTensor = torch.sum(leftTensors, dim=0)
        midTensor = torch.sum(midTensors, dim=0)
        rightTensor = torch.sum(rightTensors, dim=0)
        tensors = [leftTensor, midTensor, rightTensor]
        input_ = torch.cat(tensors).type(torch.Tensor)  #make default tensor type for network
        f = self.f_m_c(input_)
        return f

    '''
    mentions:python list of all mentions
    RETURNS:Tensor of f_m_c per mention'''

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

    def exp_bracketss(self, m_i: Mention, m_j: Mention):
        f_i = self.perform_fmc(m_i)
        f_j = self.perform_fmc(m_j)
        y = f_i.T
        y = torch.matmul(y, self.D).reshape(SETTINGS.k, SETTINGS.d)  # mulmul not dot for non-1D dotting
        y = torch.matmul(y, f_j)
        x = y / np.math.sqrt(SETTINGS.d)
        return torch.exp(x)

    '''
    m and mbar from t-1
    m & mbar: [i][j][arg]'''
    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf
    def lbp_iteration_individual(self, mbar, mentions, i, psis_i, j, arg):
        # TODO ensure candidates is Gamma(i) - Gamma is the set of candidates for a mention?
        maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here?
        phis_i = self.phis(arg, i.candidates, i, j)  # TODO - check, shouldn't arg be for i, not j?
        values = psis_i
        values += phis_i
        mbarmatrix = torch.tensor(mbar)  # Convert mbar to a torch tensor #TODO - keep it as a torch tensor
        mbarslice = mbarmatrix[:, i.id]  # Slice it to a k*arb tensor
        mbarslice[j.id] = 0  # 0 out side where k == j, so it can be summed inconsequentially
        mbarsums = mbarslice.sum(dim=0)
        values += mbarsums
        if len(values) != 0:  # Prevent identity error when tensor is empty
            maxValue = values.max()
        return maxValue

    '''
    INPUT:
    psis: python array of 1D tensors (i,arb) of psi values
    '''

    def lbp_iteration_complete(self, mbar, mentions, psis):
        newmbar = {}
        for idx, i in tqdm(enumerate(mentions)):
            psis_i = psis[idx]
            newmbar[i.id] = {}
            for j in mentions:
                newmbar[i.id][j.id] = {}
                mvalues = {}
                for arg in j.candidates:
                    newmval = self.lbp_iteration_individual(mbar, mentions, i, psis_i, j, arg)
                    mvalues[arg.id] = newmval
                mvalsum = 0  # Eq 13 denominator from LBP paper
                for value in mvalues.values():
                    mvalsum += value.exp()
                for arg in j.candidates:
                    # Bar (needs softmax)
                    dampingFactor = 0.5  # delta in the paper
                    mval = mvalues[arg.id]
                    bar = mbar[i.id][j.id][arg.id].exp()
                    bar *= (1 - dampingFactor)
                    bar += dampingFactor * (mval.exp() / mvalsum)
                    bar = bar.log()
                    newmbar[i.id][j.id][arg.id] = bar
        return newmbar

    def lbp_total(self, mentions: List[Mention], f_m_cs):
        mbar = {}
        psis = []
        for (i, f_m_c) in zip(mentions, f_m_cs):
            psis_i = self.psis(i, f_m_c)
            psis.append(psis_i)
        debug("Computing initial mbar for LBP")
        for i in tqdm(mentions):
            mbar[i.id] = {}
            for j in mentions:
                mbar[i.id][j.id] = {}
                for arg in j.candidates:
                    mbar[i.id][j.id][arg.id] = torch.Tensor([0]).reshape([])  # just a scalar
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            newmbar = self.lbp_iteration_complete(mbar, mentions, psis)
            mbar = newmbar
        # Now compute ubar
        ubar = {}
        debug("Computing final ubar out the back of LBP")
        for (i_idx, (i, f_m_c)) in tqdm(enumerate(zip(mentions, f_m_cs))):
            ubar[i.id] = {}
            for arg_idx, arg in enumerate(i.candidates):
                sum = 0
                for k in mentions:
                    if k != i:
                        sum += mbar[k.id][i.id][arg.id]
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
        f_m_cs = self.perform_fmcs(mentions)
        ubar = self.lbp_total(mentions, f_m_cs)
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
