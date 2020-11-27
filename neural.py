# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
import numpy as np
import tensorflow as tf
import torch
from torch import nn

import hyperparameters
import processeddata
from datastructures import Model, Mention
from hyperparameters import SETTINGS


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model'''


def train(model: Model):  # TODO - add params
    # TODO - train neural network using ADAM

    # TODO - fill in
    pass


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
            nn.Linear(300, 100),  # TODO what dimensions?
            nn.Tanh(),
            nn.Dropout(p=SETTINGS.dropout_rate),
        )

    # local and pairwise score functions (equation 3+section 3.1)
    '''e_i:entity embedding
    B: diagonal matrix
    c_i: context'''

    def psi(self, e_i: np.ndarray, c_i, B: np.ndarray):
        f_c = self.f(c_i)
        return e_i.T.dot(B).dot(f_c)

    '''e_i,e_j:entity embeddings
    R: k diagonal matrices
    D: diagonal matrix'''

    def phi_k(self, e_i, e_j, R, k):
        return e_i.T.dot(R[k]).dot(e_j)

    def phi(self, e_i, e_j, m_i, m_j, R, D):
        sum = 0
        for k in range(0, SETTINGS.k):
            value = self.phi_k(e_i, e_j, R, k)
            value *= self.a(D, k, m_i, None, m_j, None)
            sum += value
        return sum

    def a(self, D, k, m_i, c_i, m_j, c_j):
        z_ijk = self.Z(D, m_i, c_i, m_j, c_j)
        x = self.exp_brackets(D, k, m_i, c_i, m_j, c_j)
        return x / z_ijk

    '''D: diagonal matri'''

    def Z(self, D, m_i, c_i, m_j, c_j):
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            sum = 0
            for k_prime in range(0, SETTINGS.k):
                sum += self.exp_brackets(D, k_prime, m_i, c_i, m_j, c_j)
            return sum
        else:
            raise Exception("Unimplemented normalisation method")
        # TODO ment-norm Z

    def perform_fmc(self, m_i):
        leftWords = m_i.left_context.split(" ")
        midWords = m_i.text.split(" ")
        rightWords = m_i.right_context.split(" ")
        wordEmbedding = lambda word: processeddata.wordid2embedding[
            processeddata.word2wordid.get(word, processeddata.unkwordid)]
        leftEmbeddings = map(wordEmbedding, leftWords)
        midEmbeddings = map(wordEmbedding, midWords)
        rightEmbeddings = map(wordEmbedding, rightWords)
        leftTensors = tf.convert_to_tensor(leftEmbeddings)
        midTensors = tf.convert_to_tensor(midEmbeddings)
        rightTensors = tf.convert_to_tensor(rightEmbeddings)
        leftTensor = torch.sum(leftTensors)
        midTensor = torch.sum(midTensors)
        rightTensor = torch.sum(rightTensors)
        tensors = [leftTensor, midTensor, rightTensor]
        input_ = torch.cat(tensors)
        f = self.f_m_c(input_)
        return f

    def exp_brackets(self, D, k, m_i: Mention, c_i, m_j: Mention, c_j):
        f_i = self.perform_fmc(m_i)
        f_j = self.perform_fmc(m_j)
        y = f_i.T.dot(D[k]).dot(f_j)
        x = y / np.math.sqrt(SETTINGS.d)
        return np.math.exp(x)

    '''
    m and mbar from t-1
    m & mbar: [i][j][arg]'''

    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf
    def lbp_iteration_individual(self, mbar, i, j, arg, candidates, B, R, D):
        # TODO ensure candidates is Gamma(i) - Gamma is the set of candidates for a mention?
        maxValue = 0
        max = None
        for e_prime in candidates:
            value = self.psi(i, e_prime, B)
            m_i, m_j = None, None  # Find mentions from entity???? TODO - how to do
            value += self.phi(arg, e_prime, m_i, m_j, R, D)
            mentions = []  # TODO - set of mentions
            for k in mentions:
                if k != j:
                    value += mbar[k][i][e_prime]
            if value > maxValue:
                max = e_prime
        return max

    def lbp_iteration_complete(self, mbar, candidates, B, R, D):
        newmbar = {}
        mentions = []  # TODO - all mentions
        for i in mentions:
            newmbar[i] = {}
            for j in mentions:
                newmbar[i][j] = {}
                mvalues = {}
                for arg in mentions:
                    newmval = self.lbp_iteration_individual(mbar, i, j, arg, candidates, B, R, D)
                    mvalues[arg] = newmval
                mvalsum = 0
                for value in mvalues.values():
                    mvalsum += np.exp(value)
                for arg in mentions:
                    # Bar (needs softmax)
                    dampingFactor = 0.5  # TODO - what value for delta?
                    mval = mvalues[arg]
                    bar = np.log(
                        dampingFactor * (np.exp(mval) / mvalsum) + (1 - dampingFactor) * np.exp(mbar[i][j][arg]))
                    newmbar[i][j][arg] = bar
        return newmbar

    def lbp_total(self, candidates, B, R, D):
        mbar = {}
        mentions = []  # TODO - all mentions
        for i in mentions:
            mbar[i] = {}
            for j in mentions:
                mbar[i][j] = 0
        for loopno in range(0, SETTINGS.LBP_loops):
            newmbar = self.lbp_iteration_complete(mbar, candidates, B, R, D)
            mbar = newmbar
        # Now compute ubar
        ubar = {}
        for i in mentions:
            ubar[i] = {}
            for arg in mentions:
                sum = 0
                for k in mentions:
                    if k != i:
                        sum += mbar[k][i][arg]
                u = self.psi(i, arg, B) + sum
                ubar[i][arg] = np.exp(u)
            sum = 0
            for arg in mentions:  # Gamma(mi)
                sum += ubar[i][arg]
            for arg in mentions:
                ubar[i][arg] /= sum  # Normalise
        return ubar

    def forward(self, mentions, y, candidates, B, R, D):
        ubar = self.lbp_total(candidates, B, R, D)
        for m in mentions:  # all mentions
            for e in m.candidates:  # candidate entities
                p_e_m = e.initial_prob  # input from data
                q_e_d = ubar[i][e]  # From LBP
                g = lambda x, y: None  # 2-layer NN
                p_e = g(q_e_d, p_e_m)
                #return all p_e for all m

        return x

#TODO perhaps? pdf page 4 - investigate if Rij=diag{...} actually gives poor performance
