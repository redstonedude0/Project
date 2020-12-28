import unittest

import torch
from tqdm import tqdm

import processeddata
import testdata
import utils
from neural import NeuralNet


class TestNeural(unittest.TestCase):
    def setUp(self) -> None:
        super(TestNeural, self).setUp()
        self.network = NeuralNet()
        processeddata.loadEmbeddings()
        self.testingDoc = testdata.getTestData()
    def tearDown(self) -> None:
        pass

    def test_exp_bracket_methods_equiv(self):
        raise NotImplementedError("method exp_bracketss removed")
        # test exp_brackets code is equal
        ssss = self.network.exp_bracketssss(self.testingDoc.mentions)
        count = len(self.testingDoc.mentions)
        is_ = []
        for i in range(0, count):
            js = []
            for j in range(0, count):
                m_i = self.testingDoc.mentions[i]
                m_j = self.testingDoc.mentions[j]
                ss_vals = self.network.exp_bracketss(m_i, m_j)
                js.append(ss_vals)
            is_.append(torch.stack(js))
        ss = torch.stack(is_)
        maxError = utils.maxError(ssss, ss)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_a_methods_equiv(self):
        raise NotImplementedError("method a removed")
        # test a code is equal
        ass = self.network.ass(self.testingDoc.mentions)
        count = len(self.testingDoc.mentions)
        is_ = []
        for i in range(0, count):
            js = []
            for j in range(0, count):
                m_i = self.testingDoc.mentions[i]
                m_j = self.testingDoc.mentions[j]
                ss_vals = self.network.a(m_i, m_j)
                js.append(ss_vals)
            is_.append(torch.stack(js))
        a = torch.stack(is_)
        maxError = utils.maxError(ass, a)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_phik_methods_equiv_full(self):
        # test phi_k code is equal for all candidates for 2 mentions
        m_i = self.testingDoc.mentions[0]
        m_j = self.testingDoc.mentions[1]
        ksss = self.network.phi_ksss(m_i.candidates, m_j.candidates)
        count_i = len(m_i.candidates)
        is_ = []
        for i in range(0, count_i):
            c_i = m_i.candidates[i]
            kss = self.network.phi_kss(c_i, m_j.candidates)
            is_.append(kss)
        ksss_ = torch.stack(is_)
        maxError = utils.maxError(ksss, ksss_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_phik_methods_equiv_total(self):
        # test phi_k code is equal for all mentions for first 7 candidates
        maxTotalError = 0
        count = 0
        for m_i in tqdm(self.testingDoc.mentions):
            for m_j in self.testingDoc.mentions:
                i_cands = m_i.candidates[0:7]
                j_cands = m_j.candidates[0:7]
                ksss = self.network.phi_ksss(i_cands, j_cands)
                count_i = len(i_cands)
                is_ = []
                for i in range(0, count_i):
                    c_i = i_cands[i]
                    kss = self.network.phi_kss(c_i, j_cands)
                    is_.append(kss)
                ksss_ = torch.stack(is_)
                maxError = utils.maxError(ksss, ksss_)
                maxTotalError = max(maxTotalError, maxError)
                count += 1
                # print(f"Max(Sub)Error: {maxError}")
                # self.assertTrue(maxError < 0.01)
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_phis_methods_equiv(self):
        # test phis code is equal
        fmcs = self.network.perform_fmcs(self.testingDoc.mentions)
        ass = self.network.ass(self.testingDoc.mentions, fmcs)
        maxTotalError = 0
        count = 0
        for i_idx, m_i in enumerate(self.testingDoc.mentions):
            for j_idx, m_j in enumerate(self.testingDoc.mentions):
                i_cands = m_i.candidates[0:7]
                j_cands = m_j.candidates[0:7]
                phiss = self.network.phiss(i_cands, j_cands, i_idx, j_idx, ass)
                count_i = len(i_cands)
                is_ = []
                for i in range(0, count_i):
                    c_i = i_cands[i]
                    kss = self.network.phis(c_i, j_cands, i_idx, j_idx, ass)
                    is_.append(kss)
                phiss_ = torch.stack(is_)
                maxError = utils.maxError(phiss, phiss_)
                maxTotalError = max(maxTotalError, maxError)
                count += 1
                # print(f"Max(Sub)Error: {maxError}")
                # self.assertTrue(maxError < 0.01)
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_lbp(self):
        mentions = self.testingDoc.mentions
        mentions = [mentions[0]]
        mbar = torch.zeros(len(mentions), len(mentions), 7)
        i_idx = 0
        j_idx = 0
        i = mentions[i_idx]
        j = mentions[j_idx]
        i.candidates = i.candidates[0:5]
        j.candidates = j.candidates[0:6]
        # print("em",processeddata.wordid2embedding)
        fmcs = self.network.perform_fmcs(mentions)
        psis_i = self.network.psis(i, fmcs[i_idx])
        ass = self.network.ass(mentions, fmcs)
        lbps = self.network.lbp_iteration_individuals(mbar, i, i_idx, j, j_idx, psis_i, ass)
        # lbps_ = []
        # for c in j.candidates:
        #    lbp_ = self.network.lbp_iteration_individual(mbar, i, j_idx, i_idx, psis_i, ass, c)
        #    lbps_.append(lbp_)
        # lbps_ = torch.stack(lbps_)
        lbps_ = torch.tensor(
            [0.9441, 0.6720, 0.7305, 0.5730, 0.6093])  # just assert static answer is right and ensure nothing changes
        maxError = utils.maxError(lbps, lbps_)
        print("lbps", lbps)
        print("lbps_", lbps_)
        print(f"Max(Sub)Error: {maxError}")
        self.assertTrue(maxError < 0.01)
