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
from utils import debug, map_2D, map_1D, nantensor, maskedmax


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
    Compute PSI for all candidates for all mentions
    INPUT:
    mentions: 1D python list of all mentions
    fmcs: 2D tensor(n,d) fmc values
    RETURN:
    2D tensor (n,7) psi values per candidate per mention
    #TODO - mask?
    '''

    def psiss(self, mentions, fmcs):
        # c_i = m.left_context + m.right_context  # TODO - use context in f properly
        # I think f(c_i) is just the word embeddings in the 3 parts of context? needs to be a dim*1 vector?
        # f_c = self.f(c_i)#TODO - define f properly (Attention mechanism)
        fcs = fmcs  # TODO - is f_m_c equal to f_c???
        n = len(mentions)
        embeddings, masks = self.embeddings(mentions, n)
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
    mentions: 1D (n) python list of entities to compute phi_k with
    Returns: 
    5D (n_i,n_j,7,7,k) Tensor of phi_k values foreach k foreach candidate
    4D (n_i,n_j,7,7) Boolean mask Tensor for (arb_i,arb_j) masking'''

    def phi_ksssss(self, mentions):
        n = len(mentions)
        embeddings, masks = self.embeddings(mentions, n)
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
        # Make maskss (n_i,n_j,7_i,7_j)
        maskss = torch.matmul(masks.reshape([n, 1, 7, 1]).type(torch.Tensor),
                              masks.reshape([n, 1, 7]).type(torch.Tensor)).type(torch.BoolTensor)
        return valsss, maskss

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
        values = self.phi_ksss(candidates_i, candidates_j)
        values *= ass[i_idx][j_idx]
        return values.sum(dim=2)

    '''
    Calculates Phi for every candidate pair for every mention
    mentions: all mentions (1D python list)
    ass: 3D (n*n*k) matrix of a values 
    RETURN:
    4D (n_i,n_j,7_i,7_j) Tensor of phi values foreach candidate foreach i,j
    4D (n_i,n_j,7_i,7_j) Mask tensor
    '''

    def phissss(self, mentions, ass):
        n = len(mentions)
        # 5D(n_i,n_j,7_i,7_j,k) , 4D (n_i,n_j,7_i,7_j)
        values, mask = self.phi_ksssss(mentions)
        values *= ass.reshape([n, n, 1, 1, SETTINGS.k])  # broadcast along 7*7
        return values.sum(dim=4), mask

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
        torch.manual_seed(0)
        f = self.f_m_c(input_)
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

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf
    '''
    Perform LBP iteration for set of candidates for j (e_js/es)
    mbar: mbar message values (n,n,7) 3D tensor
    j,j_idx: j mention (candidates e come from)
    i,i_idx: i mention (candidates e' come from)
    psis_i: 1D tensor (arb_i) psi values for i (per candidate e')
    ass: 3D tensor (n,n,k) a values per i,j,k
    RETURNS:
    1D tensor (arb_j) of maximum message values for this i,j
    '''

    def lbp_iteration_individuals(self, mbar, i, i_idx, j, j_idx, psis_i, ass):
        # mbar intuition: mbar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # TODO maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here (when i has no candidates)?
        phis_ij = self.phiss(i.candidates, j.candidates, i_idx, j_idx, ass)  # 2d (arb_i,arb_j) tensor
        values = phis_ij  # values inside max{} brackets - Eq (10) LBP paper
        values = (values.T + psis_i).T  # broadcast (from (arb_i) to (arb_i,arb_j) tensor)
        # mbar is a 3D (n,n,7) tensor
        mbarslice = mbar[:, i_idx][:, 0:len(psis_i)]  # Slice it to a n*7 tensor then down to n*arb_i
        mbarslice[j_idx] = 0  # 0 out side where n == j, so it can be summed inconsequentially
        mbarsums = mbarslice.sum(dim=0)  # (arb_i) tensor of sums (1 for each e')
        values = (values.T + mbarsums).T  # broadcast (from (arb_i) to (arb_i,arb_j) tensor)
        #        if len(values) != 0:  # Prevent identity error when tensor is empty#TODO - how to prevent in multiple dimensions?
        maxValue = values.max(dim=0)[0]  # (arb_j) tensor of max values ([0] gets max not argmax)
        return maxValue

    '''
    Perform LBP iteration for all pairs of candidates
    mbar: mbar message values (n,n,7) 3D tensor
    mentions: python list of mentions (for j,i (e/e') mentions and candidates)
    psiss: 2D tensor (n,7) all psi values for each mention i (per candidate e')
    ass: 3D tensor (n,n,k) a values per i,j,k
    RETURNS:
    3D tensor (n,n,7_j) of maximum message values for each i,j and each candidate for j
    #TODO - add mbar mask
    '''

    def lbp_iteration_individualsss(self, mbar, mentions, psiss, ass):
        # mbar intuition: mbar[m_i][m_j][e_j] is how much m_i votes for e_j to be the candidate for m_j (at each timestep)
        # TODO maxValue = torch.Tensor([0]).reshape([])  # Default 0 to prevent errors, TODO - what do here (when i has no candidates)?
        n = len(mentions)
        phis, maskss = self.phissss(mentions, ass)  # 4d (n_i,n_j,7_i,7_j) tensor
        values = phis  # values inside max{} brackets - Eq (10) LBP paper
        values = values + psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        # mbar is a 3D (n_i,n_j,7_j) tensor
        mbarsum = mbar  # start by reading mbar as k->i beliefs
        # (n_k,n_i,7_i)
        # foreach j we will have a different sum, introduce j dimension
        mbarsum = mbarsum.repeat([n, 1, 1, 1])
        # (n_j,n_k,n_i,7_i)
        # 0 out where j=k
        cancel = 1 - torch.eye(n, n)  # (n_j,n_k) anti-identity
        cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
        mbarsum = mbarsum * cancel  # broadcast (n,n,1,1) to (n,n,n,7), setting (n,7) dim to 0 where j=k
        # (n_k,n_j,n_i,7_i)
        # sum to a (n_j,n_i,7_i) tensor of sums(over k) for each j,i,e'
        mbarsum = mbarsum.sum(dim=0).transpose(0, 1)
        # (n_i,n_j,7_i)
        values = values + mbarsum.reshape([n, n, 7, 1])  # broadcast (from (n_i,n_j,7_i,1) to (n_i,n_j,7_i,7_j) tensor)
        #        if len(values) != 0:  # Prevent identity error when tensor is empty#TODO - how to prevent in multiple dimensions?
        # Masked max
        maxValue = maskedmax(values, maskss, dim=2)[0]  # (n_i,n_j,7_j) tensor of max values ([0] gets max not argmax)
        return maxValue

    '''
    INPUT:
    psis: python array of 1D tensors (i,arb) of psi values
    '''

    def lbp_iteration_complete(self, mbar, mentions, psis, ass):
        newmbar = mbar.clone()
        for i_idx, i in tqdm(enumerate(mentions)):
            psis_i = psis[i_idx]
            for j_idx, j in enumerate(mentions):
                mvalues_ = self.lbp_iteration_individuals(mbar, i, i_idx, j, j_idx, psis_i, ass)
                mvalues = {}
                for arg, newmval in zip(j.candidates, mvalues_):
                    mvalues[arg.id] = newmval
                mvalsum = 0  # Eq 13 denominator from LBP paper
                for value in mvalues.values():
                    mvalsum += value.exp()
                for arg_idx, arg in enumerate(j.candidates):
                    # Bar (needs softmax)
                    dampingFactor = 0.5  # delta in the paper
                    mval = mvalues[arg.id]
                    bar = mbar[i_idx][j_idx][arg_idx].exp()
                    bar *= (1 - dampingFactor)
                    bar += dampingFactor * (mval.exp() / mvalsum)
                    bar = bar.log()
                    newmbar[i_idx][j_idx][arg_idx] = bar
        return newmbar

    def lbp_iteration_complete_new(self, mbar, mentions, psiss, ass):
        newmbar = mbar.clone()
        for i_idx, i in tqdm(enumerate(mentions)):
            for j_idx, j in enumerate(mentions):
                mvalues = self.lbp_iteration_individualsss(mbar, mentions, psiss, ass)[i_idx][j_idx]
                softmaxdenom = mvalues.exp().sum()  # Eq 11 softmax denominator from LBP paper

                # Do Eq 11 (old mbars + mvalues to new mbars)
                dampingFactor = 0.5  # delta in the paper
                bars = mbar[i_idx][j_idx].exp()
                bars *= (1 - dampingFactor)
                otherbit = dampingFactor * (mvalues.exp() / softmaxdenom)
                # add padding
                otherbit = torch.cat([otherbit, nantensor(7 - len(otherbit))])
                bars += otherbit
                bars = bars.log()
                newmbar[i_idx][j_idx] = bars
        return newmbar

    def lbp_total(self, mentions: List[Mention], f_m_cs, ass):
        mbar = {}
        psiss = self.psiss(mentions, f_m_cs)
        # Note: Should be i*j*arb but arb dependent so i*j*7 but unused cells will be 0 and trimmed
        debug("Computing initial mbar for LBP")
        mbar = torch.zeros(len(mentions), len(mentions), 7)
        debug("Now doing LBP Loops")
        for loopno in range(0, SETTINGS.LBP_loops):
            print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.lbp_iteration_complete_new(mbar, mentions, psiss, ass)
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
                u = psiss[i_idx][arg_idx] + sum
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
        self.psiss(mentions, f_m_cs)
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
