import math
import unittest

import torch
from tqdm import tqdm

import processeddata
import testdata
import utils
from datastructures import Candidate
from hyperparameters import SETTINGS
from neural import NeuralNet


class TestNeural(unittest.TestCase):
    def setUp(self) -> None:
        super(TestNeural, self).setUp()
        self.network = NeuralNet()
        processeddata.loadEmbeddings()
        self.testingDoc = testdata.getTestData()
        # at most 7 cands per mention in doc
        for m in self.testingDoc.mentions:
            m.candidates = m.candidates[0:7]
        self.testingDoc2 = testdata.getTestData()
        # Rand candidates
        torch.manual_seed(0)
        n = len(self.testingDoc2.mentions)
        rands = torch.rand(n) * 7
        for m, n in zip(self.testingDoc2.mentions, rands):
            m.candidates = m.candidates[0:math.floor(n)]
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
                expectedMask = expectedMask.type(torch.BoolTensor)
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
                expectedMask = expectedMask.type(torch.BoolTensor)
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
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        prev_msgs = torch.zeros(n, 7, n)
        import torch.nn.functional as F
        for _ in range(10):
            mask = 1 - torch.eye(n)
            SCOREPART = lbp_inputs.permute(0, 2, 1, 3)
            ent_ent_votes = SCOREPART + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            prev_msgs = msgs

        # compute marginal belief
        mask = torch.eye(n)
        ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
        ent_scores = F.softmax(ent_scores, dim=1)
        ubar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("ubar_", ubar_)
        maxError = utils.maxError(ubar, ubar_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_accuracy_minimal(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)

        SETTINGS.LBP_loops = 1  # no loops
        ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        prev_msgs = torch.zeros(n, 7, n)
        import torch.nn.functional as F
        for _ in range(SETTINGS.LBP_loops):
            mask = 1 - torch.eye(n)
            # lbp_inputs is [n_i][n_j][7_i][7_j]
            SCOREPART = lbp_inputs.permute(0, 2, 1, 3)
            SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
            # SCOREPARY is [j][7_J][i][7_i]
            ent_ent_votes = SCOREPART + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            prev_msgs = msgs

        # compute marginal belief
        mask = torch.eye(n)
        ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
        ent_scores = F.softmax(ent_scores, dim=1)
        ubar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("ubar_", ubar_)
        maxError = utils.maxError(ubar, ubar_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_accuracy_mvals(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)

        prev_msgs = torch.zeros(n, 7, n)
        # prev_msgs is [n_j][7_j][n_i]
        mbar = prev_msgs.permute(2, 0, 1).clone()
        # mbar is [n_i][n_j][7_j]
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        # lbp[i][j][e_i][e_j]
        #        SCOREPART = lbp_inputs.permute(0, 2, 1, 3)
        # [i][e_i][j][e_j]
        SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
        # ACTIVE[j][e_j][i][e_i]
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans
        ####MY CODE COPY ACROSS
        if True:
            # mbar is a 3D (n_i,n_j,7_j) tensor
            # (n_k,n_i,7_i)
            # foreach j we will have a different sum, introduce j dimension
            mbarsum = mbar.repeat([n, 1, 1, 1])
            # (n_j,n_k,n_i,7_i)
            # 0 out where j=k (do not want j->i(e'), set to 0 for all i,e')
            cancel = 1 - torch.eye(n)  # (n_j,n_k) anti-identity
            cancel = cancel.reshape([n, n, 1, 1])  # add extra dimensions
            mbarsum = mbarsum * cancel  # broadcast (n,n,1,1) to (n,n,n,7), setting (n,7) dim to 0 where j=k
            # (n_j,n_k,n_i,7_i)
            # sum to a (n_i,n_j,7_i) tensor of sums(over k) for each j,i,e'
            mbarsum = utils.smartsum(mbarsum, 1).transpose(0, 1)

            # lbp_inputs is phi+psi values, add to the mbar sums to get the values in the max brackets
            values = SCOREPART.permute(1, 3, 0, 2)  # broadcast (from (n_i,7_i,n_j,1) to (n_i,7_i,n_j,7_j) tensor)
            # TODO CHANGE1 - broadcast across all i candidates, not j candidates
            values = values
            mvals = utils.smartmax(values, dim=1)  # (n_i,n_j,7_j) tensor of max values
            mvals = mvals.permute(2, 1, 0)
            # print("mbar",maxValue)
        ###END MY CODE COPY ACROSS

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        if True:
            cancel_ = 1 - torch.eye(n)
            ent_ent_votes = SCOREPART
            msgs, _ = torch.max(ent_ent_votes,
                                dim=3)  # TODO - appears the paper has an error? summing over wrong dimension?
            mvals_ = msgs
        #            msgs, _ = torch.max(ent_ent_votes, dim=3)
        #            mvals_ = msgs.permute(0, 2, 1)
        # ORIGINAL CODE RESUME
        mvals_ = mvals_.permute(2, 0, 1)

        print("mvals", mvals)
        print("mvals_", mvals_)
        print("mvals", mvals.shape)
        print("mvals_", mvals_.shape)
        print("DIFF", mvals - mvals_)
        maxError = utils.maxError(mvals, mvals_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_accuracy_mvals_fixed(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)

        mbar = torch.randn(n, n, 7)
        prev_msgs = mbar.permute(1, 2, 0).clone()
        # prev_msgs is [n_j][7_j][n_i]
        # mbar is [n_i][n_j][7_j]
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        # lbp[i][j][e_i][e_j]
        SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
        # ACTIVE[j][e_j][i][e_i] (summing over e_i)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans
        ####MY CODE COPY ACROSS
        if True:
            mvals = self.network.lbp_iteration_complete(mbar, masks, n, lbp_inputs)

        ###END MY CODE COPY ACROSS

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        if True:
            mask = 1 - torch.eye(n)
            ent_ent_votes = SCOREPART + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (torch.nn.functional.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            prev_msgs = msgs
            # msgs is [j][e_j][i]
            mvals_ = msgs.permute(2, 0, 1)  # to i,j,ej
        # ORIGINAL CODE RESUME

        print("mvals", mvals)
        print("mvals_", mvals_)
        print("mvals", mvals.shape)
        print("mvals_", mvals_.shape)
        print("DIFF", mvals - mvals_)
        maxError = utils.maxError(mvals, mvals_)
        print(f"MaxError: {maxError}")
        self.assertTrue(maxError < 0.01)

    def test_lbp_accuracy_mvals_fixed2(self):
        # Test LBP compared to original papers implementation
        mentions = self.testingDoc.mentions
        # Enforce 7 cands (add extra UNK cands - MRN code relies on exactly 7 cands existing
        for m in mentions:
            if len(m.candidates) != 7:
                cands = m.candidates
                extracand = Candidate(-1, 0.5, "#UNK#")
                extracands = [extracand for i in range(0, 7 - len(cands))]
                m.candidates = cands + extracands
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.ass(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)

        mbar = torch.randn(n, n, 7)
        prev_msgs = mbar.permute(1, 2, 0).clone()
        # prev_msgs is [n_j][7_j][n_i]
        # mbar is [n_i][n_j][7_j]
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        # lbp[i][j][e_i][e_j]
        SCOREPART = lbp_inputs.permute(1, 3, 0, 2)
        # ACTIVE[j][e_j][i][e_i] (summing over e_i)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans
        mvals = self.network.lbp_iteration_mvaluesss(mbar, n, lbp_inputs)
        ####MY CODE COPY ACROSS
        if True:
            expmvals = mvals.exp()
            utils.setMaskBroadcastable(expmvals, ~masks.reshape([1, n, 7]), 0)  # nans are 0
            softmaxdenoms = utils.smartsum(expmvals, dim=2)  # Eq 11 softmax denominator from LBP paper

            # Do Eq 11 (old mbars + mvalues to new mbars)
            dampingFactor = SETTINGS.dropout_rate  # delta in the paper
            newmbar = mbar.exp()
            newmbar *= (1 - dampingFactor)
            #        print("X1 ([0.25-]0.5)",newmbar)
            #        print("expm",expmvals)
            otherbit = dampingFactor * (expmvals / softmaxdenoms.reshape([n, n, 1]))  # broadcast (n,n) to (n,n,7)

            print("mine", (expmvals / softmaxdenoms.reshape([n, n, 1])))
            print("answer", torch.nn.functional.softmax(mvals, dim=2))

            #        print("X2 (0-0.5)",otherbit)
            newmbar2 = newmbar + otherbit
            #        print("X3 ([0.25-]0.5-1.0)",newmbar)
            utils.setMaskBroadcastable(newmbar2, ~masks.reshape([1, n, 7]),
                                       1)  # 'nan's become 0 after log (broadcast (1,n,7) to (n,n,7))
            mbarvals = newmbar2.log()
        #            mbarvals = self.network.lbp_iteration_complete(mbar,masks,n,lbp_inputs)
        #        print("X4 (-0.69 - 0)",newmbar)
            # newmbar

        ###END MY CODE COPY ACROSS

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        if True:
            msgs = mvals.permute(1, 2, 0)  # i,j,ej->j,ej,i
            msgs = (torch.nn.functional.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            #            prelogtheirs = torch.nn.functional.softmax(mvals.permute(1,2,0), dim=1).mul(SETTINGS.dropout_rate)
            #            prelogours = otherbit
            #            print("at",torch.nn.functional.softmax(mvals.permute(1,2,0), dim=1).permute(2,0,1))
            #            print("ao",(expmvals / softmaxdenoms.reshape([n, n, 1])))
            #            print("ao2",torch.nn.functional.softmax(mvals,dim=2))
            # prev_msgs = msgs
            # msgs is [j][e_j][i]
            mvals_ = msgs.permute(2, 0, 1)  # to i,j,ej
        # ORIGINAL CODE RESUME

        print("mbarvals", mbarvals)
        print("mvals_", mvals_)
        print("mbarvals", mbarvals.shape)
        print("mvals_", mvals_.shape)
        print("DIFF", mbarvals - mvals_)
        diff = mbarvals - mvals_
        print("diff", (diff != 0).nonzero())
        print("diff", diff[diff != 0])
        print("example", mbarvals[0][11], mvals_[0][11])
        print("src", mvals[0][11], mbar[0][11])
        print("truth", (mvals[0][11].softmax(0) * 0.3 + 0.7).log())
        print("truth", (mvals[0][11].softmax(0) * 0.3 + mbar.exp()[0][11] * 0.7).log())
        #        print("truth", (mvals.softmax(2)*0.3+0.7).log()[0][11])
        #        print("truth", (mvals.permute(1, 2, 0).softmax(1)*0.3+0.7).log().permute(2,0,1)[0][11])
        #        print("truth", (torch.nn.functional.softmax(mvals.permute(1, 2, 0), dim=1)*0.3+0.7).log().permute(2,0,1)[0][11])
        #        print("truth", (torch.nn.functional.softmax(mvals.permute(1, 2, 0), dim=1).mul(SETTINGS.dropout_rate)+0.7).log().permute(2,0,1)[0][11])
        #        print("truth", (torch.nn.functional.softmax(mvals.permute(1, 2, 0), dim=1).mul(SETTINGS.dropout_rate)+prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log().permute(2,0,1)[0][11])
        sumError = utils.sumError(mbarvals, mvals_)
        print(f"MaxError: {sumError}")
        self.assertTrue(sumError < 0.01)
