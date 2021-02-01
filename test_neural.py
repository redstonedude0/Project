import math
import unittest

import torch
from tqdm import tqdm

import neural
import processeddata
import testdata
import utils
from datastructures import Candidate, Dataset
from hyperparameters import SETTINGS
from neural import NeuralNet


class TestNeural(unittest.TestCase):
    def setUp(self) -> None:
        super(TestNeural, self).setUp()
        self.network = NeuralNet()
        processeddata.loadEmbeddings()

        # Trimmed cands
        self.testingDoc = testdata.getTestData()
        # at most 7 cands per mention in doc
        for m in self.testingDoc.mentions:
            m.candidates = m.candidates[0:7]

        # Rand candidates
        self.testingDoc2 = testdata.getTestData()
        torch.manual_seed(0)
        n = len(self.testingDoc2.mentions)
        rands = torch.rand(n) * 7
        for m, n in zip(self.testingDoc2.mentions, rands):
            m.candidates = m.candidates[0:math.floor(n)]

        # 7 cands (pad with unks)
        self.testingDoc3 = testdata.getTestData()
        extracand = Candidate(-1, 0.5, "#UNK#")
        for m in self.testingDoc3.mentions:
            if len(m.candidates) != 7:
                cands = m.candidates[0:7]  # Trim if needed
                extracands = [extracand for i in range(0, 7 - len(cands))]  # Expand if needed
                m.candidates = cands + extracands

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
        raise NotImplementedError("phi_kss removed")
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
        raise NotImplementedError("phi_kss removed")
        # test phi_k code is equal for all mentions for first 7 candidates
        maxTotalError = 0
        count = 0
        for m_i in tqdm(self.testingDoc.mentions):
            for m_j in self.testingDoc.mentions:
                i_cands = m_i.candidates
                j_cands = m_j.candidates
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
        raise NotImplementedError("Methos phis removed")
        # test phis code is equal
        fmcs = self.network.perform_fmcs(self.testingDoc.mentions)
        ass = self.network.ass(self.testingDoc.mentions, fmcs)
        maxTotalError = 0
        count = 0
        for i_idx, m_i in enumerate(self.testingDoc.mentions):
            for j_idx, m_j in enumerate(self.testingDoc.mentions):
                i_cands = m_i.candidates
                j_cands = m_j.candidates
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

    def test_lbp_individual(self):
        raise NotImplementedError("lbp_iteration_individuals removed")
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

    def test_lbp_complete(self):
        raise NotImplementedError("Changed _new signature, no longer needed")
        mentions = self.testingDoc.mentions
        mbar = torch.zeros(len(mentions), len(mentions), 7)
        # print("em",processeddata.wordid2embedding)
        fmcs = self.network.perform_fmcs(mentions)
        psis = [self.network.psis(m, fmcs[m_idx]) for m_idx, m in enumerate(mentions)]
        ass = self.network.ass(mentions, fmcs)
        lbp = self.network.lbp_iteration_complete_new(mbar, mentions, psis, ass)
        lbp_ = self.network.lbp_iteration_complete(mbar, mentions, psis, ass)
        maxError = utils.maxError(lbp, lbp_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_phik_equiv_5D(self):
        raise NotImplementedError("phi_ksss removed")
        mentions = self.testingDoc.mentions
        phisss, maskss = self.network.phi_ksssss(mentions)
        print("PHISSS", phisss[11][0])
        maxTotalError = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            for j_idx, j in enumerate(mentions):
                phis_ = self.network.phi_ksss(i.candidates, j.candidates)
                # Check the error between them only as far as the arbs of phis_
                arb_i, arb_j, k = phis_.shape
                phisss_sub = phisss[i_idx][j_idx].narrow(0, 0, arb_i).narrow(1, 0, arb_j)
                maxError = utils.maxError(phis_, phisss_sub)
                print(f"Max(Sub)Error: {maxError}")
                self.assertTrue(maxError < 0.01)
                maxTotalError = max(maxTotalError, maxError)
                count += 1
                # Check maskss
                mask = maskss[i_idx][j_idx]
                expectedMask = torch.zeros([7, 7])
                horizontalMask = torch.zeros([7])
                horizontalMask[0:arb_j] = 1
                expectedMask[0:arb_i] = horizontalMask
                expectedMask = expectedMask.to(torch.bool)
                self.assertTrue(mask.equal(expectedMask))

                # print(f"Max(Sub)Error: {maxError}")
                # self.assertTrue(maxError < 0.01)
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_phis_equiv_5D(self):
        raise NotImplementedError("phiss removed")
        mentions = self.testingDoc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(mentions, fmcs)
        phisss, maskss = self.network.phissss(mentions, ass)
        maxTotalError = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            for j_idx, j in enumerate(mentions):
                phis_ = self.network.phiss(i.candidates, j.candidates, i_idx, j_idx, ass)
                # Check the error between them only as far as the arbs of phis_
                arb_i, arb_j = phis_.shape
                phisss_sub = phisss[i_idx][j_idx].narrow(0, 0, arb_i).narrow(1, 0, arb_j)
                maxError = utils.maxError(phis_, phisss_sub)
                print(f"Max(Sub)Error: {maxError}")
                self.assertTrue(maxError < 0.01)
                maxTotalError = max(maxTotalError, maxError)
                count += 1
                # Check maskss
                mask = maskss[i_idx][j_idx]
                expectedMask = torch.zeros([7, 7])
                horizontalMask = torch.zeros([7])
                horizontalMask[0:arb_j] = 1
                expectedMask[0:arb_i] = horizontalMask
                expectedMask = expectedMask.to(torch.bool)
                self.assertTrue(mask.equal(expectedMask))

                # print(f"Max(Sub)Error: {maxError}")
                # self.assertTrue(maxError < 0.01)
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_psis_equiv(self):
        raise NotImplementedError("psis removed")
        mentions = self.testingDoc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        psiss = self.network.psiss(mentions, fmcs)
        print("psiss1", psiss[11])
        print("psiss2", self.network.psis(mentions[11], fmcs[11]))
        maxTotalError = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            psis_ = self.network.psis(i, fmcs[i_idx])
            # Check the error between them only as far as the arb of psis_
            arb_i = psis_.shape[0]  # need [0] to explicitely cast from torch.size to int
            psiss_sub = psiss[i_idx].narrow(0, 0, arb_i)
            maxError = utils.maxError(psis_, psiss_sub)
            print(f"Max(Sub)Error: {maxError}")
            self.assertTrue(maxError < 0.01)
            maxTotalError = max(maxTotalError, maxError)
            count += 1
            # Check masks
            # TODO (if add masks)
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_lbp_indiv_equiv(self):
        raise NotImplementedError("lbp_iteration_individuals removed")
        mentions = self.testingDoc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(mentions, fmcs)
        psiss = self.network.psiss(mentions, fmcs)
        torch.manual_seed(0)
        mbar = torch.rand([len(mentions), len(mentions), 7])
        mbarnew, masks = self.network.lbp_iteration_individualsss(mbar, mentions, psiss, ass)
        maxTotalError = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            psis_i = psiss[i_idx][0:len(i.candidates)]
            for j_idx, j in enumerate(mentions):
                mbarnew_ = self.network.lbp_iteration_individuals(mbar, i, i_idx, j, j_idx, psis_i, ass)
                # Check the error between them only as far as the arbs of mbarnew_
                arb_j = mbarnew_.shape[0]
                mbarnew_sub = mbarnew[i_idx][j_idx].narrow(0, 0, arb_j)
                maxError = utils.maxError(mbarnew_sub, mbarnew_)
                if maxError > 1:
                    print(f"Max(Sub)Error: {maxError}")
                    print("at", i_idx, j_idx)
                    print("comp", mbarnew[i_idx][j_idx], mbarnew_)
                    print("comp", i_idx, j_idx)
                #                    self.assertTrue(maxError < 0.01)
                maxTotalError = max(maxTotalError, maxError)
                count += 1
        print(f"MaxError: {maxTotalError} (of {count} pairs)")
        self.assertTrue(maxTotalError < 0.01)

    def test_lbp_compl_equiv(self):
        raise NotImplementedError("lbp_iteration_complete removed")
        mentions = self.testingDoc.mentions
        embs, maskss = self.network.embeddings(mentions, len(mentions))
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(mentions, fmcs)
        psiss = self.network.psiss(mentions, fmcs)
        # random mbar for better testing
        mbar = torch.rand([len(mentions), len(mentions), 7])
        mbarnew = self.network.lbp_iteration_complete_new(mbar, mentions, psiss, ass)
        psis = []
        for i_idx, i in enumerate(mentions):
            psis_ = psiss[i_idx][0:len(i.candidates)]
            psis.append(psis_)
        mbarnew_ = self.network.lbp_iteration_complete(mbar, mentions, psis, ass)
        maxError = utils.maxErrorMasked(mbarnew, mbarnew_, maskss)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_total_equiv(self):
        raise NotImplementedError("lbp_total_old removed")
        mentions = self.testingDoc.mentions
        embs, maskss = self.network.embeddings(mentions, len(mentions))
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        ubar = self.network.lbp_total(mentions, fmcs, ass)
        ubar_ = self.network.lbp_total_old(mentions, fmcs, ass)
        n = len(mentions)
        ubar_2 = torch.zeros([n, 7])
        for i_idx, i in enumerate(mentions):
            for arg_idx, arg in enumerate(i.candidates):
                ubar_2[i_idx][arg_idx] = ubar_[i.id][arg.id]
        maxError = utils.maxError(ubar, ubar_2)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_neural_consistency(self):
        saving = False
        output = self.network.forward(self.testingDoc2)
        if saving:
            torch.save(output, "test_neural_consistency.pt")
            print(output)
            raise Exception("Saving consistency map, failing test...")
        else:
            load = torch.load("test_neural_consistency.pt")
            load = load.reshape([30, 7])
            maxError = utils.maxError(output, load)
            print(f"MaxError: {maxError}")
            print(output)
            self.assertTrue(maxError == 0)

    def test_lbp_accuracy(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        # MY CODE
        if True:
            ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)
        #
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = SCOREPART + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs

            # compute marginal belief
            mask = 1 - torch.eye(n)
            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
            ent_scores = F.softmax(ent_scores, dim=1)
            ubar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("ubar_", ubar_)
        maxError = utils.maxError(ubar_, ubar)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc]
        modeller.candidatePadding()
        self.testingDoc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        # MY CODE
        if True:
            ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)

        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = SCOREPART + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs

            # compute marginal belief
            mask = 1 - torch.eye(n)
            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
            ent_scores = F.softmax(ent_scores, dim=1)
            ubar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("ubar_", ubar_)
        print("diff",ubar-ubar_)
        maxError = utils.maxError(ubar_, ubar)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_originalimplementationbounds(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        #
#        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
#        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        #MINE
        mbar = torch.zeros(n,n,7)
        for loopno in range(0, SETTINGS.LBP_loops):
            print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
            newmbar = self.network.lbp_iteration_complete(mbar, masks, n, lbp_inputs)
            mbar = newmbar
            print("SUM",mbar.exp().sum(dim=2))

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = SCOREPART + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs
                print("MAX",msgs.max())
                print("MIN",msgs.min())
                print("SUM",msgs.exp().sum(dim=1))

            # compute marginal belief
            mask = 1 - torch.eye(n)
            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
            ent_scores = F.softmax(ent_scores, dim=1)
            ubar_ = ent_scores
        # ORIGINAL CODE RESUME
        print("ubar_", ubar_)

    def test_lbp_nanconsistency(self):
        mentions = self.testingDoc.mentions
        n = len(mentions)
        SETTINGS.allow_nans = True
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)

        SETTINGS.allow_nans = False
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        ubar_ = self.network.lbp_total(n, masks, psiss, lbp_inputs)

        print("ubar", ubar)
        print("ubar_", ubar_)
        maxError = utils.maxError(ubar, ubar_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbpit_nanconsistency(self):
        mentions = self.testingDoc.mentions
        n = len(mentions)
        SETTINGS.allow_nans = True
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        mbar = torch.zeros(n, n, 7).to(SETTINGS.device)
        mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
        mbar *= mbar_mask
        mvals = self.network.lbp_iteration_mvaluesss(mbar,masks,n,lbp_inputs)

        SETTINGS.allow_nans = False
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        mbar = torch.zeros(n, n, 7).to(SETTINGS.device)
        mbar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            mbar_mask[mbar_mask == 0] = nan  # make nan not 0
        mbar *= mbar_mask
        mvals_ = self.network.lbp_iteration_mvaluesss(mbar,masks,n,lbp_inputs)

        print("mvals", mvals)
        print("mvals_", mvals_)
        maxError = utils.maxError(mvals, mvals_)
        diff = mvals-mvals_
        diff[diff!=diff] = 0#make nans zero
#        print("Where diff?",diff.nonzero())#index of where to investigate (11,_ here)
        print("diff",diff[11,0,:])
        print("mvals",mvals[11,0,:])
        print("mvals_",mvals_[11,0,:])
#        print(len(mentions[11].candidates),"candidates")
#        print(phis[11][0])
#        print(psiss[11])
#        print(mbar[:,11,0])
#        print(mbar[:,11,1])
#        print("lbps",lbp_inputs[11,0,:,:])
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)


    def test_fwd_nan_consistency(self):
        SETTINGS.allow_nans = True
        results = self.network.forward(self.testingDoc)
        SETTINGS.allow_nans = False
        results_ = self.network.forward(self.testingDoc)

        print("results", results)
        print("results_", results_)
        maxError = utils.maxError(results, results_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)
        # where results is nan - results_ is 0
        self.assertTrue((results != results).equal(results_ == 0))

    def test_TEMP_investigatedoc(self):
        import datasets
        import modeller
        SETTINGS.dataset = datasets.loadDataset("aida_train.csv")
        modeller.candidateSelection()
        offender = SETTINGS.dataset.documents[9]
        out = self.network.forward(offender)
        print(out[5])
        print(offender.mentions[5].candidates)
        # TODO - delete test (investigation done)

    def test_loss_consistency(self):
        import datasets
        import modeller
        SETTINGS.dataset = datasets.loadDataset("aida_train.csv")
        modeller.candidateSelection()
        doc1 = SETTINGS.dataset.documents[0]
        doc2 = SETTINGS.dataset.documents[1]
        net1 = NeuralNet()
        net2 = NeuralNet()
        net3 = NeuralNet()

        def method1(net):
            # Do cumulative loss
            opt1 = torch.optim.Adam(net.parameters(), lr=SETTINGS.learning_rate_initial)
            opt1.zero_grad()
            loss = neural.loss_regularisation(net.R, net.D)
            out1 = net.forward(doc1)
            losspart = neural.loss_document(doc1, out1)
            loss += losspart
            out2 = net.forward(doc2)
            losspart = neural.loss_document(doc2, out2)
            loss += losspart
            loss.backward()
            opt1.step()

        def method2(net):
            # Do individual loss
            opt2 = torch.optim.Adam(net.parameters(), lr=SETTINGS.learning_rate_initial)
            opt2.zero_grad()
            loss = neural.loss_regularisation(net.R, net.D)
            loss.backward()
            out1 = net.forward(doc1)
            loss = neural.loss_document(doc1, out1)
            loss.backward()
            out2 = net.forward(doc2)
            loss = neural.loss_document(doc2, out2)
            loss.backward()
            opt2.step()

        method1(net1)
        method1(net2)
        method2(net3)

        print("Parameter dump:")
        for (n1, param1), (n2, param2), (n3, param3) in zip(net1.named_parameters(), net2.named_parameters(),
                                                            net3.named_parameters()):
            if n1 != n2 or n2 != n3:
                raise Exception("Parameter names didn't match - this should never happen.")
            print(f" Parameter {n1}:")
            print(f" Reflexive diff: {param1 - param2}")
            print(f" Comparative diff: {param1 - param3}")
        print("Assessing consistency...")
        for (n1, param1), (n2, param2), (n3, param3) in zip(net1.named_parameters(), net2.named_parameters(),
                                                            net3.named_parameters()):
            sumError1 = utils.sumError(param1, param2)
            sumError2 = utils.sumError(param1, param3)
            sumError3 = utils.sumError(param2, param3)
            print(f" Parameter {n1} Errors: SELF[1-2]({sumError1}) OTHER[1-3]({sumError2}) OTHER[2-3]({sumError3})")


########################################################
########################################################
########################################################
########################################################

    def test_lbpmvals_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc]
        modeller.candidatePadding()
        self.testingDoc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initialMbar = torch.randn(n,n,7)
        # MY CODE
        if True:
            mvalues = self.network.lbp_iteration_mvaluesss(initialMbar.clone(), masks, n, lbp_inputs.clone())

        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            #Ours: n votes for n to have candidate 7
            #Theirs:n has candidate 7 as voted for by n
            prev_msgs = initialMbar.permute(1,2,0)
            import torch.nn.functional as F
            mask = 1 - torch.eye(n)
            SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
            ent_ent_votes = SCOREPART + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            mvalues_ = msgs.permute(2,0,1)
        # ORIGINAL CODE RESUME

        print("mvalues",mvalues)
        print("mvalues_",mvalues_)
        print("diff",mvalues-mvalues_)
        maxError = utils.maxError(mvalues_, mvalues)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbpmbarloop_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc]
        modeller.candidatePadding()
        self.testingDoc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initialMbar = torch.randn(n,n,7)
        # MY CODE
        if True:
            mbar = initialMbar.clone()
            for loopno in range(0, SETTINGS.LBP_loops):
                print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
                newmbar = self.network.lbp_iteration_complete(mbar, masks, n, lbp_inputs.clone())
                mbar = newmbar
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            #Ours: n votes for n to have candidate 7
            #Theirs:n has candidate 7 as voted for by n

            prev_msgs = initialMbar.permute(1,2,0)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = SCOREPART + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs

            mbar_ = prev_msgs.permute(2,0,1)
            # compute marginal belief
#            mask = 1 - torch.eye(n)
#            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
#            ent_scores = F.softmax(ent_scores, dim=1)
#            ubar_ = ent_scores


        # ORIGINAL CODE RESUME

        print("mbar",mbar)
        print("mbar_",mbar_)
        print("diff",mbar-mbar_)
        maxError = utils.maxError(mbar_, mbar)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)


    def test_lbpmbar_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc]
        modeller.candidatePadding()
        self.testingDoc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initialMbar = torch.randn(n,n,7)
        # MY CODE
        if True:
            mbar = initialMbar.clone()
            newmbar = self.network.lbp_iteration_complete(mbar, masks, n, lbp_inputs.clone())
            mbar = newmbar
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            #Ours: n votes for n to have candidate 7
            #Theirs:n has candidate 7 as voted for by n

            prev_msgs = initialMbar.permute(1,2,0)
            import torch.nn.functional as F
            mask = 1 - torch.eye(n)
            SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
            ent_ent_votes = SCOREPART + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            prev_msgs = msgs

            mbar_ = prev_msgs.permute(2,0,1)
            # compute marginal belief
#            mask = 1 - torch.eye(n)
#            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
#            ent_scores = F.softmax(ent_scores, dim=1)
#            ubar_ = ent_scores


        # ORIGINAL CODE RESUME

        print("mbar",mbar)
        print("mbar_",mbar_)
        print("diff",mbar-mbar_)
        maxError = utils.maxError(mbar_, mbar)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)



    def test_lbpmbar_accuracy2b(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testingDoc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc]
        modeller.candidatePadding()
        self.testingDoc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initialMbar = torch.randn(n,n,7)
        initialMvalues = self.network.lbp_iteration_mvaluesss(initialMbar.clone(), masks, n, lbp_inputs.clone())
        # MY CODE
        if True:
            mvalues = initialMvalues.clone()
            # LBPEq11 - softmax mvalues
            # softmax invariant under translation, translate to around 0 to reduce float errors
            from utils import normalise_avgToZero_rowWise,setMaskBroadcastable,smartsum
            #normalise_avgToZero_rowWise(mvalues, masks.reshape([1, n, 7]), dim=2)
            expmvals = mvalues.exp()
            expmvals = expmvals.clone()  # clone to prevent autograd errors (in-place modification next)
            softmaxdenoms = expmvals.sum( dim=2)  # Eq 11 softmax denominator from LBP paper
            softmaxmvalues = expmvals / softmaxdenoms.reshape([n, n, 1])  # broadcast (n,n) to (n,n,7)
            #Micro rounding errors!!!

        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        #NOTE: MRN code is not designed for non-7 candidates
        if True:
            #Ours: n votes for n to have candidate 7
            #Theirs:n has candidate 7 as voted for by n

            msgs = initialMvalues.permute(1,2,0)
            import torch.nn.functional as F
            softmaxmvalues_ = F.softmax(msgs, dim=1)#
            softmaxmvalues_ = softmaxmvalues_.permute(2,0,1)
            # compute marginal belief
#            mask = 1 - torch.eye(n)
#            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
#            ent_scores = F.softmax(ent_scores, dim=1)
#            ubar_ = ent_scores


        # ORIGINAL CODE RESUME

        #print("eq1",eq1)
        #print("eq1",eq1)
        print("diff1",softmaxmvalues-softmaxmvalues_)
        maxError1 = utils.maxError(softmaxmvalues_,softmaxmvalues)
        print(f"MaxError1: {maxError1}")
        self.assertTrue(maxError1 == 0)


    def test_lbp_invariants(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc2.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testingDoc2]
#        modeller.candidatePadding()
        self.testingDoc2 = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initialMbar = torch.randn(n,n,7)
        #Eq 11 invariants:
        if True:
            # Softmax of m should sum to 1 for each i,j
            #  Except: sum should be 0 when no cands
            mvalues = self.network.lbp_iteration_mvaluesss(initialMbar.clone(), masks, n, lbp_inputs.clone())
            from utils import normalise_avgToZero_rowWise, setMaskBroadcastable, smartsum
            normalise_avgToZero_rowWise(mvalues, masks.reshape([1, n, 7]), dim=2)
            expmvals = mvalues.exp()
            expmvals = expmvals.clone()  # clone to prevent autograd errors (in-place modification next)
            setMaskBroadcastable(expmvals, ~masks.reshape([1, n, 7]), 0)  # nans are 0
            softmaxdenoms = smartsum(expmvals, dim=2)  # Eq 11 softmax denominator from LBP paper
            softmaxmvalues = expmvals / softmaxdenoms.reshape([n, n, 1])  # broadcast (n,n) to (n,n,7)
            for i in range(0,n):
                for j in range(0,n):
                    softmaxline = softmaxmvalues[i][j]
                    maskline = masks[j]
                    entries = int(maskline.to(torch.float).sum().item())
                    softmaxline = softmaxline[0:entries]
                    linesum = softmaxline.sum()
                    expectedValue = int(masks[j].sum() >= 1)
                    print(linesum,expectedValue)
                    self.assertTrue((linesum-expectedValue).abs() < 5e-07)
        #Invariants for sum of exp mbars:
        #Sum(logmbareqns) = X
        #log(mbar0eq)+log(mbar1eq)+...=X
        #log(PI(mbareq)) =
        #log(mbareq(n))=mbar(n) in -inf,0
        #mbareq(n) in 0,1
        #softmax(e) in 0,1 #By defn
        #exp(mbar) in 0,1 #By assumption+initial

        #Round 0: sum(exp(mbar)) = ncands
        #Round 1: sum(exp(mbar)) = sum(delta*softmax)+sum(1-delta*expmbar))
        #                        = delta*1+ (1-delta)*ncands
        #Round 2:                = delta*1+(1-delta)*(delta+(1-delta)*ncands)
        #R1 = 0.3+0.7*ncands
        #R2 = 0.3+0.7*(0.3+0.7*ncands) = 0.51+0.49*ncands
        #R3 = 0.3+0.7*(0.51+0.49*ncands) = 0.657+0.343*ncands
        if True:
            mbar = torch.zeros(n,n,7)
            desiredSums = masks.to(torch.float).sum(dim=1)
            desiredSums_new = desiredSums.clone()
            for loop in range(0,10):
                print("----NEWLOOP----")
                desiredSums = desiredSums_new.clone()
                newmbar = self.network.lbp_iteration_complete(mbar,masks,n,lbp_inputs)
                mbar = newmbar
                mbarexp = mbar.exp()
                #Assert invariant
                for i in range(0,n):
                    for j in range(0,n):
                        mbarexpline = mbarexp[i][j]
                        maskline = masks[j]
                        entries = int(maskline.to(torch.float).sum().item())
                        mbarexpline = mbarexpline[0:entries]
                        linesum = mbarexpline.sum()
                        desiredsum = desiredSums[j]
                        desiredsum = 0.3+0.7*desiredsum
                        if masks[j].sum() == 0:
                            #Empty cands, should sum to 0 insted
                            desiredsum = 0
                        desiredSums_new[j] = desiredsum#save for next iter
                        print(linesum,desiredsum)
                        print("diff",(linesum-desiredsum).abs())
                        self.assertTrue((linesum-desiredsum).abs() < 1e-06)
            #In the end each line sums to roughly ~2.0084 (+- e-06)
            print("FINAL MBAR")
            print(mbar)