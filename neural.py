# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
from typing import List

import numpy as np
import tensorflow as tf
import torch
from torch import nn

import hyperparameters
import processeddata
from datastructures import Model, Mention, Candidate, Document
from hyperparameters import SETTINGS


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model'''


def train(model: Model):  # TODO - add params
    model.neuralNet: NeuralNet
    print("Model output:")
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
            nn.Linear(300, 100),  # TODO what dimensions?
            nn.Tanh(),
            nn.Dropout(p=SETTINGS.dropout_rate),
        )
        # TODO - set default parameter values properly
        self.register_parameter("B", torch.nn.Parameter(torch.diag(torch.ones(100))))
        self.register_parameter("R", torch.nn.Parameter(torch.diag(torch.ones(100))))
        self.register_parameter("D", torch.nn.Parameter(torch.diag(torch.ones(100))))

    # local and pairwise score functions (equation 3+section 3.1)
    '''e_i:entity
    B: diagonal matrix
    c_i: context'''

    def psi(self, e: Candidate, m: Mention, B: np.ndarray):
        c_i = m.left_context + m.right_context  # TODO - use context in f properly
        # I think f(c_i) is just the word embeddings in the 3 parts of context? needs to be a dim*1 vector?
        # f_c = self.f(c_i)#TODO - define f properly (Attention mechanism)
        f_c = self.perform_fmc(m)  # TODO - is f_m_c equal to f_c???
        return e.entEmbedding().T.dot(B).dot(f_c)

    '''e_i,e_j:entity embeddings
    R: k diagonal matrices
    D: diagonal matrix'''

    def phi_k(self, e_i, e_j, R, k):
        return e_i.T.dot(R[k]).dot(e_j)

    def phi(self, e_i, e_j, m_i, m_j, R, D):
        sum = 0
        for k in range(0, SETTINGS.k):
            value = self.phi_k(e_i, e_j, R, k)
            value *= self.a(D, k, m_i, m_j)
            sum += value
        return sum

    def a(self, D, k, m_i, m_j):
        z_ijk = self.Z(D, m_i, m_j)
        x = self.exp_brackets(D, k, m_i, m_j)
        return x / z_ijk

    '''D: diagonal matri'''

    def Z(self, D, m_i, m_j):
        if SETTINGS.normalisation == hyperparameters.NormalisationMethod.RelNorm:
            sum = 0
            for k_prime in range(0, SETTINGS.k):
                sum += self.exp_brackets(D, k_prime, m_i, m_j)
            return sum
        else:
            raise Exception("Unimplemented normalisation method")
        # TODO ment-norm Z

    def perform_fmc(self, m_i):
        leftWords = m_i.left_context.split(" ")
        midWords = m_i.text.split(" ")
        rightWords = m_i.right_context.split(" ")
        wordEmbedding = lambda word: processeddata.wordid2embedding[
            processeddata.word2wordid.get(word,
                                          processeddata.unkwordid)].tolist()  # convert np array to list so tf can convert to tensor
        leftEmbeddings = list(map(wordEmbedding, leftWords))
        midEmbeddings = list(map(wordEmbedding, midWords))
        rightEmbeddings = list(map(wordEmbedding, rightWords))
        print("EMBEDDINGS ARE", leftEmbeddings)
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

    def exp_brackets(self, D, k, m_i: Mention, m_j: Mention):
        f_i = self.perform_fmc(m_i)
        f_j = self.perform_fmc(m_j)
        y = f_i.T.dot(D[k]).dot(f_j)
        x = y / np.math.sqrt(SETTINGS.d)
        return np.math.exp(x)

    '''
    m and mbar from t-1
    m & mbar: [i][j][arg]'''
    # LBP FROM https://arxiv.org/pdf/1704.04920.pdf
    def lbp_iteration_individual(self, mbar, mentions, i, j, arg, B, R, D):
        # TODO ensure candidates is Gamma(i) - Gamma is the set of candidates for a mention?
        maxValue = 0
        max = None
        for e_prime in i.candidates:
            value = self.psi(e_prime, i, B)
            value += self.phi(arg, e_prime, i, j, R, D)
            for k in mentions:
                if k != j:
                    value += mbar[k.id][i.id][e_prime.id]
            if value > maxValue:
                max = e_prime
        return max

    def lbp_iteration_complete(self, mbar, mentions, B, R, D):
        newmbar = {}
        for i in mentions:
            newmbar[i.id] = {}
            for j in mentions:
                newmbar[i.id][j.id] = {}
                mvalues = {}
                for arg in j.candidates:
                    newmval = self.lbp_iteration_individual(mbar, mentions, i, j, arg, B, R, D)
                    mvalues[arg.id] = newmval
                mvalsum = 0
                for value in mvalues:
                    mvalsum += np.exp(value)
                for arg in j.candidates:
                    # Bar (needs softmax)
                    dampingFactor = 0.5  # delta in the paper
                    mval = mvalues[arg.id]
                    bar = np.log(
                        dampingFactor * (np.exp(mval) / mvalsum) + (1 - dampingFactor) * np.exp(
                            mbar[i.id][j.id][arg.id]))
                    newmbar[i.id][j.id][arg.id] = bar
        return newmbar

    def lbp_total(self, mentions: List[Mention], B, R, D):
        mbar = {}
        for i in mentions:
            mbar[i.id] = {}
            for j in mentions:
                mbar[i.id][j.id] = {}
                for arg in j.candidates:
                    mbar[i.id][j.id][arg.id] = 0
        for loopno in range(0, SETTINGS.LBP_loops):
            newmbar = self.lbp_iteration_complete(mbar, mentions, B, R, D)
            mbar = newmbar
        # Now compute ubar
        ubar = {}
        for i in mentions:
            ubar[i.id] = {}
            for arg in i.candidates:
                sum = 0
                for k in mentions:
                    if k != i:
                        sum += mbar[k.id][i.id][arg.id]
                u = self.psi(arg, i, B) + sum
                ubar[i.id][arg.id] = np.exp(u)
            sum = 0
            for arg in i.candidates:  # Gamma(mi)
                sum += ubar[i.id][arg.id]
            for arg in i.candidates:
                ubar[i.id][arg.id] /= sum  # Normalise
        return ubar

    def forward(self, document: Document):
        mentions = document.mentions
        ubar = self.lbp_total(mentions, self.B, self.R, self.D)
        m: Mention
        p = {}
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
