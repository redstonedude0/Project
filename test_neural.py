import math
import unittest

import torch
from tqdm import tqdm

import datasets
import modeller
import neural
import processeddata
import testdata
import utils
from datastructures import Candidate, Dataset, Model, EvalHistory
from hyperparameters import SETTINGS, NormalisationMethod
from neural import NeuralNet


class TestNeural(unittest.TestCase):
    """
    non-pep8 name, required by unittest
    """

    def setUp(self) -> None:
        super(TestNeural, self).setUp()
        self.network = NeuralNet()
        processeddata.load_embeddings()

        # Trimmed cands
        self.testing_doc = testdata.get_test_data()
        # at most 7 cands per mention in doc
        for m in self.testing_doc.mentions:
            m.candidates = m.candidates[0:7]

        # Rand candidates
        self.testing_doc2 = testdata.get_test_data()
        torch.manual_seed(0)
        n = len(self.testing_doc2.mentions)
        rands = torch.rand(n) * 7
        for m, n in zip(self.testing_doc2.mentions, rands):
            m.candidates = m.candidates[0:math.floor(n)]

        # 8 cands (pad with unks)
        self.testing_doc3 = testdata.get_test_data()
        SETTINGS.n_cands_ctx = 4
        SETTINGS.n_cands_pem = 4
        SETTINGS.n_cands = 8
        dataset = Dataset()
        dataset.documents = [self.testing_doc3]
        # self.testing_doc3 = None#Cause error for now, want to disale candidate sel for testing
        # modeller.candidate_selection(dataset,"tDoc3",True)

    """
    non-pep8 name, required by unittest
    """

    def tearDown(self) -> None:
        pass

    def test_exp_bracket_methods_equiv(self):
        raise NotImplementedError("method exp_bracketss removed")
        # test exp_brackets code is equal
        ssss = self.network.exp_bracketssss(self.testing_doc.mentions)
        count = len(self.testing_doc.mentions)
        is_ = []
        for i in range(0, count):
            js = []
            for j in range(0, count):
                m_i = self.testing_doc.mentions[i]
                m_j = self.testing_doc.mentions[j]
                ss_vals = self.network.exp_bracketss(m_i, m_j)
                js.append(ss_vals)
            is_.append(torch.stack(js))
        ss = torch.stack(is_)
        max_error = utils.max_error(ssss, ss)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_a_methods_equiv(self):
        raise NotImplementedError("method a removed")
        # test a code is equal
        ass = self.network.asss(self.testing_doc.mentions)
        count = len(self.testing_doc.mentions)
        is_ = []
        for i in range(0, count):
            js = []
            for j in range(0, count):
                m_i = self.testing_doc.mentions[i]
                m_j = self.testing_doc.mentions[j]
                ss_vals = self.network.a(m_i, m_j)
                js.append(ss_vals)
            is_.append(torch.stack(js))
        a = torch.stack(is_)
        max_error = utils.max_error(ass, a)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_phik_methods_equiv_full(self):
        raise NotImplementedError("phi_kss removed")
        # test phi_k code is equal for all candidates for 2 mentions
        m_i = self.testing_doc.mentions[0]
        m_j = self.testing_doc.mentions[1]
        ksss = self.network.phi_ksss(m_i.candidates, m_j.candidates)
        count_i = len(m_i.candidates)
        is_ = []
        for i in range(0, count_i):
            c_i = m_i.candidates[i]
            kss = self.network.phi_kss(c_i, m_j.candidates)
            is_.append(kss)
        ksss_ = torch.stack(is_)
        max_error = utils.max_error(ksss, ksss_)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_phik_methods_equiv_total(self):
        raise NotImplementedError("phi_kss removed")
        # test phi_k code is equal for all mentions for first 7 candidates
        max_total_error = 0
        count = 0
        for m_i in tqdm(self.testing_doc.mentions):
            for m_j in self.testing_doc.mentions:
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
                max_error = utils.max_error(ksss, ksss_)
                max_total_error = max(max_total_error, max_error)
                count += 1
                # print(f"Max(Sub)Error: {max_error}")
                # self.assertTrue(max_error < 0.01)
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_phis_methods_equiv(self):
        raise NotImplementedError("Methos phis removed")
        # test phis code is equal
        fmcs = self.network.perform_fmcs(self.testing_doc.mentions)
        asss = self.network.asss(self.testing_doc.mentions, fmcs)
        max_total_error = 0
        count = 0
        for i_idx, m_i in enumerate(self.testing_doc.mentions):
            for j_idx, m_j in enumerate(self.testing_doc.mentions):
                i_cands = m_i.candidates
                j_cands = m_j.candidates
                phiss = self.network.phiss(i_cands, j_cands, i_idx, j_idx, asss)
                count_i = len(i_cands)
                is_ = []
                for i in range(0, count_i):
                    c_i = i_cands[i]
                    kss = self.network.phis(c_i, j_cands, i_idx, j_idx, asss)
                    is_.append(kss)
                phiss_ = torch.stack(is_)
                max_error = utils.max_error(phiss, phiss_)
                max_total_error = max(max_total_error, max_error)
                count += 1
                # print(f"Max(Sub)Error: {max_error}")
                # self.assertTrue(max_error < 0.01)
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_lbp_individual(self):
        raise NotImplementedError("lbp_iteration_individuals removed")
        mentions = self.testing_doc.mentions
        mentions = [mentions[0]]
        m_bar = torch.zeros(len(mentions), len(mentions), 7)
        i_idx = 0
        j_idx = 0
        i = mentions[i_idx]
        j = mentions[j_idx]
        i.candidates = i.candidates[0:5]
        j.candidates = j.candidates[0:6]
        # print("em",processeddata.word_id_to_embedding)
        fmcs = self.network.perform_fmcs(mentions)
        psis_i = self.network.psis(i, fmcs[i_idx])
        ass = self.network.asss(mentions, fmcs)
        lbps = self.network.lbp_iteration_individuals(m_bar, i, i_idx, j, j_idx, psis_i, ass)
        # lbps_ = []
        # for c in j.candidates:
        #    lbp_ = self.network.lbp_iteration_individual(m_bar, i, j_idx, i_idx, psis_i, asss, c)
        #    lbps_.append(lbp_)
        # lbps_ = torch.stack(lbps_)
        lbps_ = torch.tensor(
            [0.9441, 0.6720, 0.7305, 0.5730, 0.6093])  # just assert static answer is right and ensure nothing changes
        max_error = utils.max_error(lbps, lbps_)
        print("lbps", lbps)
        print("lbps_", lbps_)
        print(f"Max(Sub)Error: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_complete(self):
        raise NotImplementedError("Changed _new signature, no longer needed")
        mentions = self.testing_doc.mentions
        m_bar = torch.zeros(len(mentions), len(mentions), 7)
        # print("em",processeddata.word_id_to_embedding)
        fmcs = self.network.perform_fmcs(mentions)
        psis = [self.network.psis(m, fmcs[m_idx]) for m_idx, m in enumerate(mentions)]
        ass = self.network.asss(mentions, fmcs)
        lbp = self.network.lbp_iteration_complete_new(m_bar, mentions, psis, ass)
        lbp_ = self.network.lbp_iteration_complete(m_bar, mentions, psis, ass)
        max_error = utils.max_error(lbp, lbp_)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_phik_equiv_5D(self):
        raise NotImplementedError("phi_ksss removed")
        mentions = self.testing_doc.mentions
        phisss, maskss = self.network.phi_ksssss(mentions)
        print("PHISSS", phisss[11][0])
        max_total_error = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            for j_idx, j in enumerate(mentions):
                phis_ = self.network.phi_ksss(i.candidates, j.candidates)
                # Check the error between them only as far as the arbs of phis_
                arb_i, arb_j, k = phis_.shape
                phisss_sub = phisss[i_idx][j_idx].narrow(0, 0, arb_i).narrow(1, 0, arb_j)
                max_error = utils.max_error(phis_, phisss_sub)
                print(f"Max(Sub)Error: {max_error}")
                self.assertTrue(max_error < 0.01)
                max_total_error = max(max_total_error, max_error)
                count += 1
                # Check maskss
                mask = maskss[i_idx][j_idx]
                expected_mask = torch.zeros([7, 7])
                horizontal_mask = torch.zeros([7])
                horizontal_mask[0:arb_j] = 1
                expected_mask[0:arb_i] = horizontal_mask
                expected_mask = expected_mask.to(torch.bool)
                self.assertTrue(mask.equal(expected_mask))

                # print(f"Max(Sub)Error: {max_error}")
                # self.assertTrue(max_error < 0.01)
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_phis_equiv_5D(self):
        raise NotImplementedError("phiss removed")
        mentions = self.testing_doc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(mentions, fmcs)
        phisss, maskss = self.network.phissss(mentions, ass)
        max_total_error = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            for j_idx, j in enumerate(mentions):
                phis_ = self.network.phiss(i.candidates, j.candidates, i_idx, j_idx, ass)
                # Check the error between them only as far as the arbs of phis_
                arb_i, arb_j = phis_.shape
                phisss_sub = phisss[i_idx][j_idx].narrow(0, 0, arb_i).narrow(1, 0, arb_j)
                max_error = utils.max_error(phis_, phisss_sub)
                print(f"Max(Sub)Error: {max_error}")
                self.assertTrue(max_error < 0.01)
                max_total_error = max(max_total_error, max_error)
                count += 1
                # Check maskss
                mask = maskss[i_idx][j_idx]
                expected_mask = torch.zeros([7, 7])
                horizontal_mask = torch.zeros([7])
                horizontal_mask[0:arb_j] = 1
                expected_mask[0:arb_i] = horizontal_mask
                expected_mask = expected_mask.to(torch.bool)
                self.assertTrue(mask.equal(expected_mask))

                # print(f"Max(Sub)Error: {max_error}")
                # self.assertTrue(max_error < 0.01)
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_psis_equiv(self):
        raise NotImplementedError("psis removed")
        mentions = self.testing_doc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        psiss = self.network.psiss(mentions, fmcs)
        print("psiss1", psiss[11])
        print("psiss2", self.network.psis(mentions[11], fmcs[11]))
        max_total_error = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            psis_ = self.network.psis(i, fmcs[i_idx])
            # Check the error between them only as far as the arb of psis_
            arb_i = psis_.shape[0]  # need [0] to explicitely cast from torch.size to int
            psiss_sub = psiss[i_idx].narrow(0, 0, arb_i)
            max_error = utils.max_error(psis_, psiss_sub)
            print(f"Max(Sub)Error: {max_error}")
            self.assertTrue(max_error < 0.01)
            max_total_error = max(max_total_error, max_error)
            count += 1
            # Check masks
            # TODO (if add masks)
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_lbp_indiv_equiv(self):
        raise NotImplementedError("lbp_iteration_individuals removed")
        mentions = self.testing_doc.mentions
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(mentions, fmcs)
        psiss = self.network.psiss(mentions, fmcs)
        torch.manual_seed(0)
        m_bar = torch.rand([len(mentions), len(mentions), 7])
        m_bar_new, masks = self.network.lbp_iteration_individualsss(m_bar, mentions, psiss, ass)
        max_total_error = 0
        count = 0
        for i_idx, i in enumerate(mentions):
            psis_i = psiss[i_idx][0:len(i.candidates)]
            for j_idx, j in enumerate(mentions):
                mbarnew_ = self.network.lbp_iteration_individuals(m_bar, i, i_idx, j, j_idx, psis_i, ass)
                # Check the error between them only as far as the arbs of mbarnew_
                arb_j = mbarnew_.shape[0]
                mbarnew_sub = m_bar_new[i_idx][j_idx].narrow(0, 0, arb_j)
                max_error = utils.max_error(mbarnew_sub, mbarnew_)
                if max_error > 1:
                    print(f"Max(Sub)Error: {max_error}")
                    print("at", i_idx, j_idx)
                    print("comp", m_bar_new[i_idx][j_idx], mbarnew_)
                    print("comp", i_idx, j_idx)
                #                    self.assertTrue(max_error < 0.01)
                max_total_error = max(max_total_error, max_error)
                count += 1
        print(f"MaxError: {max_total_error} (of {count} pairs)")
        self.assertTrue(max_total_error < 0.01)

    def test_lbp_compl_equiv(self):
        raise NotImplementedError("lbp_iteration_complete removed")
        mentions = self.testing_doc.mentions
        embs, maskss = self.network.embeddings(mentions, len(mentions))
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(mentions, fmcs)
        psiss = self.network.psiss(mentions, fmcs)
        # random m_bar for better testing
        m_bar = torch.rand([len(mentions), len(mentions), 7])
        m_bar_new = self.network.lbp_iteration_complete_new(m_bar, mentions, psiss, ass)
        psis = []
        for i_idx, i in enumerate(mentions):
            psis_ = psiss[i_idx][0:len(i.candidates)]
            psis.append(psis_)
        m_bar_new_ = self.network.lbp_iteration_complete(m_bar, mentions, psis, ass)
        max_error = utils.max_error_masked(m_bar_new, m_bar_new_, maskss)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_total_equiv(self):
        raise NotImplementedError("lbp_total_old removed")
        mentions = self.testing_doc.mentions
        embs, maskss = self.network.embeddings(mentions, len(mentions))
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        u_bar = self.network.lbp_total(mentions, fmcs, ass)
        u_bar_ = self.network.lbp_total_old(mentions, fmcs, ass)
        n = len(mentions)
        u_bar_2 = torch.zeros([n, 7])
        for i_idx, i in enumerate(mentions):
            for arg_idx, arg in enumerate(i.candidates):
                u_bar_2[i_idx][arg_idx] = u_bar_[i.id][arg.id]
        max_error = utils.max_error(u_bar, u_bar_2)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_neural_consistency(self):
        saving = False
        output = self.network.forward(self.testing_doc2)
        if saving:
            torch.save(output, "test_neural_consistency.pt")
            print(output)
            raise Exception("Saving consistency map, failing test...")
        else:
            load = torch.load("test_neural_consistency.pt")
            load = load.reshape([30, 7])
            max_error = utils.max_error(output, load)
            print(f"MaxError: {max_error}")
            print(output)
            self.assertTrue(max_error == 0)

    def test_lbp_accuracy(self):
        # Test LBP compared to original papers implementation
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        # MY CODE
        if True:
            ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)
        #
        utils.set_mask_broadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        utils.set_mask_broadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                score_part = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = score_part + \
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
            u_bar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("ubar_", u_bar_)
        max_error = utils.max_error(u_bar_, ubar)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc]
        modeller.candidatePadding()
        self.testing_doc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
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
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                score_part = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = score_part + \
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
            u_bar_ = ent_scores
        # ORIGINAL CODE RESUME

        print("ubar", ubar)
        print("u_bar_", u_bar_)
        print("diff", ubar - u_bar_)
        max_error = utils.max_error(u_bar_, ubar)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_original_implementation_bounds(self):
        # Test LBP compared to original papers implementation
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        #
        #        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([1, n, 1, 7]), 0)
        #        utils.setMaskBroadcastable(lbp_inputs, ~masks.reshape([n, 1, 7, 1]), 0)
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # MINE
        m_bar = torch.zeros(n, n, 7)
        for loop_i in range(0, SETTINGS.LBP_loops):
            print(f"Doing loop {loop_i + 1}/{SETTINGS.LBP_loops}")
            new_m_bar = self.network.lbp_iteration_complete(m_bar, masks, n, lbp_inputs)
            m_bar = new_m_bar
            print("SUM", m_bar.exp().sum(dim=2))

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            prev_msgs = torch.zeros(n, 7, n)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                score_part = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = score_part + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs
                print("MAX", msgs.max())
                print("MIN", msgs.min())
                print("SUM", msgs.exp().sum(dim=1))

            # compute marginal belief
            mask = 1 - torch.eye(n)
            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
            ent_scores = F.softmax(ent_scores, dim=1)
            u_bar_ = ent_scores
        # ORIGINAL CODE RESUME
        print("u_bar_", u_bar_)

    def test_lbp_nan_consistency(self):
        mentions = self.testing_doc.mentions
        n = len(mentions)
        SETTINGS.allow_nans = True
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        ubar = self.network.lbp_total(n, masks, psiss, lbp_inputs)

        SETTINGS.allow_nans = False
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        u_bar_ = self.network.lbp_total(n, masks, psiss, lbp_inputs)

        print("ubar", ubar)
        print("u_bar_", u_bar_)
        max_error = utils.max_error(ubar, u_bar_)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_it_nan_consistency(self):
        mentions = self.testing_doc.mentions
        n = len(mentions)
        SETTINGS.allow_nans = True
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        m_bar = torch.zeros(n, n, 7).to(SETTINGS.device)
        m_bar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            m_bar_mask[m_bar_mask == 0] = nan  # make nan not 0
        m_bar *= m_bar_mask
        m_vals = self.network.lbp_iteration_mvaluesss(m_bar, masks, n, lbp_inputs)

        SETTINGS.allow_nans = False
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        m_bar = torch.zeros(n, n, 7).to(SETTINGS.device)
        m_bar_mask = masks.repeat([n, 1, 1]).to(torch.float)  # 1 where keep,0 where nan-out
        if SETTINGS.allow_nans:
            nan = float("nan")
            m_bar_mask[m_bar_mask == 0] = nan  # make nan not 0
        m_bar *= m_bar_mask
        m_vals_ = self.network.lbp_iteration_mvaluesss(m_bar, masks, n, lbp_inputs)

        print("m_vals", m_vals)
        print("m_vals_", m_vals_)
        max_error = utils.max_error(m_vals, m_vals_)
        diff = m_vals - m_vals_
        diff[diff != diff] = 0  # make nans zero
        #        print("Where diff?",diff.nonzero())#index of where to investigate (11,_ here)
        print("diff", diff[11, 0, :])
        print("m_vals", m_vals[11, 0, :])
        print("m_vals_", m_vals_[11, 0, :])
        #        print(len(mentions[11].candidates),"candidates")
        #        print(phis[11][0])
        #        print(psiss[11])
        #        print(m_bar[:,11,0])
        #        print(m_bar[:,11,1])
        #        print("lbps",lbp_inputs[11,0,:,:])
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_fwd_nan_consistency(self):
        SETTINGS.allow_nans = True
        results = self.network.forward(self.testing_doc)
        SETTINGS.allow_nans = False
        results_ = self.network.forward(self.testing_doc)

        print("results", results)
        print("results_", results_)
        max_error = utils.max_error(results, results_)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)
        # where results is nan - results_ is 0
        self.assertTrue((results != results).equal(results_ == 0))

    def test_TEMP_investigate_doc(self):
        import datasets
        import modeller
        SETTINGS.dataset = datasets.load_dataset("aida_train.csv")
        modeller.candidate_selection()
        offender = SETTINGS.dataset.documents[9]
        out = self.network.forward(offender)
        print(out[5])
        print(offender.mentions[5].candidates)
        # TODO - delete test (investigation done)

    def test_loss_consistency(self):
        import datasets
        import modeller
        SETTINGS.dataset = datasets.load_dataset("aida_train.csv")
        modeller.candidate_selection()
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
            loss_part = neural.loss_document(doc1, out1)
            loss += loss_part
            out2 = net.forward(doc2)
            loss_part = neural.loss_document(doc2, out2)
            loss += loss_part
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
            sum_error1 = utils.sum_error(param1, param2)
            sum_error2 = utils.sum_error(param1, param3)
            sum_error3 = utils.sum_error(param2, param3)
            print(f" Parameter {n1} Errors: SELF[1-2]({sum_error1}) OTHER[1-3]({sum_error2}) OTHER[2-3]({sum_error3})")

    ########################################################
    ########################################################
    ########################################################
    ########################################################

    def test_lbp_m_vals_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc]
        modeller.candidatePadding()
        self.testing_doc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initial_m_bar = torch.randn(n, n, 7)
        # MY CODE
        if True:
            m_values = self.network.lbp_iteration_mvaluesss(initial_m_bar.clone(), masks, n, lbp_inputs.clone())

        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            # Ours: n votes for n to have candidate 7
            # Theirs:n has candidate 7 as voted for by n
            prev_msgs = initial_m_bar.permute(1, 2, 0)
            import torch.nn.functional as F
            mask = 1 - torch.eye(n)
            score_part = lbp_inputs.permute(1, 3, 0, 2)
            ent_ent_votes = score_part + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            m_values_ = msgs.permute(2, 0, 1)
        # ORIGINAL CODE RESUME

        print("m_values", m_values)
        print("m_values_", m_values_)
        print("diff", m_values - m_values_)
        max_error = utils.max_error(m_values_, m_values)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_m_bar_loop_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc]
        modeller.candidate_padding()
        self.testing_doc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initial_m_bar = torch.randn(n, n, 7)
        # MY CODE
        if True:
            m_bar = initial_m_bar.clone()
            for loopno in range(0, SETTINGS.LBP_loops):
                print(f"Doing loop {loopno + 1}/{SETTINGS.LBP_loops}")
                new_m_bar = self.network.lbp_iteration_complete(m_bar, masks, n, lbp_inputs.clone())
                m_bar = new_m_bar
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            # Ours: n votes for n to have candidate 7
            # Theirs:n has candidate 7 as voted for by n

            prev_msgs = initial_m_bar.permute(1, 2, 0)
            import torch.nn.functional as F
            for _ in range(10):
                mask = 1 - torch.eye(n)
                score_part = lbp_inputs.permute(1, 3, 0, 2)
                ent_ent_votes = score_part + \
                                torch.sum(prev_msgs.view(1, n, 7, n) *
                                          mask.view(n, 1, 1, n), dim=3) \
                                    .view(n, 1, n, 7)
                msgs, _ = torch.max(ent_ent_votes, dim=3)
                msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                        prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
                prev_msgs = msgs

            m_bar_ = prev_msgs.permute(2, 0, 1)
            # compute marginal belief
        #            mask = 1 - torch.eye(n)
        #            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
        #            ent_scores = F.softmax(ent_scores, dim=1)
        #            ubar_ = ent_scores

        # ORIGINAL CODE RESUME

        print("m_bar", m_bar)
        print("m_bar_", m_bar_)
        print("diff", m_bar - m_bar_)
        max_error = utils.max_error(m_bar_, m_bar)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_m_bar_accuracy2(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc]
        modeller.candidate_padding()
        self.testing_doc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initial_m_bar = torch.randn(n, n, 7)
        # MY CODE
        if True:
            m_bar = initial_m_bar.clone()
            new_m_bar = self.network.lbp_iteration_complete(m_bar, masks, n, lbp_inputs.clone())
            m_bar = new_m_bar
        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            # Ours: n votes for n to have candidate 7
            # Theirs:n has candidate 7 as voted for by n

            prev_msgs = initial_m_bar.permute(1, 2, 0)
            import torch.nn.functional as F
            mask = 1 - torch.eye(n)
            score_part = lbp_inputs.permute(1, 3, 0, 2)
            ent_ent_votes = score_part + \
                            torch.sum(prev_msgs.view(1, n, 7, n) *
                                      mask.view(n, 1, 1, n), dim=3) \
                                .view(n, 1, n, 7)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (F.softmax(msgs, dim=1).mul(SETTINGS.dropout_rate) +
                    prev_msgs.exp().mul(1 - SETTINGS.dropout_rate)).log()
            prev_msgs = msgs

            m_bar_ = prev_msgs.permute(2, 0, 1)
            # compute marginal belief
        #            mask = 1 - torch.eye(n)
        #            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
        #            ent_scores = F.softmax(ent_scores, dim=1)
        #            ubar_ = ent_scores

        # ORIGINAL CODE RESUME

        print("m_bar", m_bar)
        print("m_bar_", m_bar_)
        print("diff", m_bar - m_bar_)
        max_error = utils.max_error(m_bar_, m_bar)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error < 0.01)

    def test_lbp_m_bar_accuracy2b(self):
        # Test LBP compared to original papers implementation
        # Correct within 2E-05
        mentions = self.testing_doc.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc]
        modeller.candidatePadding()
        self.testing_doc = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initial_m_bar = torch.randn(n, n, 7)
        initial_m_values = self.network.lbp_iteration_mvaluesss(initial_m_bar.clone(), masks, n, lbp_inputs.clone())
        # MY CODE
        if True:
            m_values = initial_m_values.clone()
            # LBPEq11 - softmax m_values
            # softmax invariant under translation, translate to around 0 to reduce float errors
            from utils import normalise_avg_to_zero_rows, set_mask_broadcastable, smart_sum
            # normalise_avgToZero_rowWise(m_values, masks.reshape([1, n, 7]), dim=2)
            exp_m_vals = m_values.exp()
            exp_m_vals = exp_m_vals.clone()  # clone to prevent autograd errors (in-place modification next)
            softmax_denoms = exp_m_vals.sum(dim=2)  # Eq 11 softmax denominator from LBP paper
            softmax_m_values = exp_m_vals / softmax_denoms.reshape([n, n, 1])  # broadcast (n,n) to (n,n,7)
            # Micro rounding errors!!!

        input_sum = lbp_inputs.sum()
        self.assertTrue(input_sum == input_sum)  # to ensure no nans

        # CODE FROM MULRELNEL ORIGINAL PAPER (*modified)
        # NOT ORIGINAL CODE
        # NOTE: MRN code is not designed for non-7 candidates
        if True:
            # Ours: n votes for n to have candidate 7
            # Theirs:n has candidate 7 as voted for by n

            msgs = initial_m_values.permute(1, 2, 0)
            import torch.nn.functional as F
            softmax_m_values_ = F.softmax(msgs, dim=1)  #
            softmax_m_values_ = softmax_m_values_.permute(2, 0, 1)
            # compute marginal belief
        #            mask = 1 - torch.eye(n)
        #            ent_scores = psiss * 1 + torch.sum(prev_msgs * mask.view(n, 1, n), dim=2)
        #            ent_scores = F.softmax(ent_scores, dim=1)
        #            ubar_ = ent_scores

        # ORIGINAL CODE RESUME

        # print("eq1",eq1)
        # print("eq1",eq1)
        print("diff1", softmax_m_values - softmax_m_values_)
        max_error1 = utils.max_error(softmax_m_values_, softmax_m_values)
        print(f"MaxError1: {max_error1}")
        self.assertTrue(max_error1 == 0)

    def test_lbp_invariants(self):
        # Test LBP compared to original papers implementation
        mentions = self.testing_doc2.mentions
        SETTINGS.allow_nans = False
        import modeller
        SETTINGS.dataset = Dataset()
        SETTINGS.dataset.documents = [self.testing_doc2]
        #        modeller.candidatePadding()
        self.testing_doc2 = SETTINGS.dataset.documents[0]
        n = len(mentions)
        embeddings, masks = self.network.embeddings(mentions, n)
        fmcs = self.network.perform_fmcs(mentions)
        ass = self.network.asss(fmcs, n)
        phis = self.network.phissss(n, embeddings, ass)
        psiss = self.network.psiss(n, embeddings, fmcs)
        lbp_inputs = phis  # values inside max{} brackets - Eq (10) LBP paper
        lbp_inputs += psiss.reshape([n, 1, 7, 1])  # broadcast (from (n_i,7_i) to (n_i,n_j,7_i,7_j) tensor)
        initial_m_bar = torch.randn(n, n, 7)
        # Eq 11 invariants:
        if True:
            # Softmax of m should sum to 1 for each i,j
            #  Except: sum should be 0 when no cands
            m_values = self.network.lbp_iteration_mvaluesss(initial_m_bar.clone(), masks, n, lbp_inputs.clone())
            from utils import normalise_avg_to_zero_rows, set_mask_broadcastable, smart_sum
            normalise_avg_to_zero_rows(m_values, masks.reshape([1, n, 7]), dim=2)
            exp_m_values = m_values.exp()
            exp_m_values = exp_m_values.clone()  # clone to prevent autograd errors (in-place modification next)
            set_mask_broadcastable(exp_m_values, ~masks.reshape([1, n, 7]), 0)  # nans are 0
            softmax_denoms = smart_sum(exp_m_values, dim=2)  # Eq 11 softmax denominator from LBP paper
            softmax_m_values = exp_m_values / softmax_denoms.reshape([n, n, 1])  # broadcast (n,n) to (n,n,7)
            for i in range(0, n):
                for j in range(0, n):
                    softmax_line = softmax_m_values[i][j]
                    mask_line = masks[j]
                    entries = int(mask_line.to(torch.float).sum().item())
                    softmax_line = softmax_line[0:entries]
                    line_sum = softmax_line.sum()
                    expected_value = int(masks[j].sum() >= 1)
                    print(line_sum, expected_value)
                    self.assertTrue((line_sum - expected_value).abs() < 5e-07)
        # Invariants for sum of exp mbars:
        # Sum(logmbareqns) = X
        # log(mbar0eq)+log(mbar1eq)+...=X
        # log(PI(mbareq)) =
        # log(mbareq(n))=m_bar(n) in -inf,0
        # mbareq(n) in 0,1
        # softmax(e) in 0,1 #By defn
        # exp(m_bar) in 0,1 #By assumption+initial

        # Round 0: sum(exp(m_bar)) = ncands
        # Round 1: sum(exp(m_bar)) = sum(delta*softmax)+sum(1-delta*expmbar))
        #                        = delta*1+ (1-delta)*ncands
        # Round 2:                = delta*1+(1-delta)*(delta+(1-delta)*ncands)
        # R1 = 0.3+0.7*ncands
        # R2 = 0.3+0.7*(0.3+0.7*ncands) = 0.51+0.49*ncands
        # R3 = 0.3+0.7*(0.51+0.49*ncands) = 0.657+0.343*ncands
        if True:
            m_bar = torch.zeros(n, n, 7)
            desired_sums = masks.to(torch.float).sum(dim=1)
            desired_sums_new = desired_sums.clone()
            for loop in range(0, 10):
                print("----NEWLOOP----")
                desired_sums = desired_sums_new.clone()
                new_m_bar = self.network.lbp_iteration_complete(m_bar, masks, n, lbp_inputs)
                m_bar = new_m_bar
                m_bar_exp = m_bar.exp()
                # Assert invariant
                for i in range(0, n):
                    for j in range(0, n):
                        m_bar_exp_line = m_bar_exp[i][j]
                        mask_line = masks[j]
                        entries = int(mask_line.to(torch.float).sum().item())
                        m_bar_exp_line = m_bar_exp_line[0:entries]
                        line_sum = m_bar_exp_line.sum()
                        desired_sum = desired_sums[j]
                        desired_sum = 0.3 + 0.7 * desired_sum
                        if masks[j].sum() == 0:
                            # Empty cands, should sum to 0 insted
                            desired_sum = 0
                        desired_sums_new[j] = desired_sum  # save for next iter
                        print(line_sum, desired_sum)
                        print("diff", (line_sum - desired_sum).abs())
                        self.assertTrue((line_sum - desired_sum).abs() < 1e-06)
            # In the end each line sums to roughly ~2.0084 (+- e-06)
            print("FINAL MBAR")
            print(m_bar)

    def test_psi_consistency(self):
        SETTINGS.allow_nans = False
        #        SETTINGS.dataset = Dataset()
        #        SETTINGS.dataset.documents = [self.testing_doc3]
        # SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv","AIDA/aida_train.txt")
        # SETTINGS.dataset_eval = datasets.load_dataset("aida_testA.csv","AIDA/testa_testb_aggregate_original")
        #        doc = self.testing_doc3
        #        mentions= doc.mentions

        SETTINGS.dataset = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        modeller.candidate_selection(SETTINGS.dataset, "TEST_train", True)
        doc = SETTINGS.dataset.documents[0]
        mentions = doc.mentions

        #####THEIRS (removed CUDA dependency)
        import sys
        sys.path.insert(1, "mulrel-nel-master/")
        import nel.local_ctx_att_ranker as local_ctx_att_ranker
        import nel.utils as nelutils
        #        local_ctx_att_ranker.DEBUG = {}
        #        neural.DEBUG = {}
        voca_emb_dir = "/home/harrison/Documents/project/data/generated/embeddings/word_ent_embs/"  # SETTINGS.data_dir_embeddings
        word_voca, word_embeddings = nelutils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                             voca_emb_dir + 'word_embeddings.npy')
        entity_voca, entity_embeddings = nelutils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                                 voca_emb_dir + 'entity_embeddings.npy')
        config = {'hid_dims': 100,
                  'tok_top_n': 25,
                  'margin': 0.01,
                  'emb_dims': entity_embeddings.shape[1],
                  'word_voca': word_voca,
                  'entity_voca': entity_voca,
                  'freeze_embs': True,
                  'word_embeddings': word_embeddings,
                  'entity_embeddings': entity_embeddings,
                  'ctx_window': 100,
                  'n_cands_before_rank': 30
                  }
        print("CONFIG", config)
        model = local_ctx_att_ranker.LocalCtxAttRanker(config)
        # p_e_m is None (i think, from mulrel_ranker), although is a FloatTensor from ed_ranker, is never passed to local_ctx.
        # Compute tokens using their code
        contexts = []
        for m in doc.mentions:
            lctx = m.left_context.strip().split()
            lctx_ids = [word_voca.get_id(t) for t in lctx if nelutils.is_important_word(t)]
            lctx_ids = [tid for tid in lctx_ids if
                        tid != word_voca.unk_id]  # TODO - assuming self.prerank_model.word_voca is word_voca and not snd_voca (99% sure I'm right)
            lctx_ids = lctx_ids[max(0, len(lctx_ids) - config['ctx_window'] // 2):]

            rctx = m.right_context.strip().split()
            rctx_ids = [word_voca.get_id(t) for t in rctx if nelutils.is_important_word(t)]
            rctx_ids = [tid for tid in rctx_ids if tid != word_voca.unk_id]
            rctx_ids = rctx_ids[:min(len(rctx_ids), config['ctx_window'] // 2)]

            contexts.append((lctx_ids, rctx_ids))

        # token_ids is left@right context, or [unk], LEFT padded with unks, to equal length per mention, long tensor
        # token_mask is 1 iff exists for token_ids, including the initial [unk] if no ctx
        token_ids = [l + r
                     if len(l) + len(r) > 0
                     else [word_voca.unk_id]
                     for (l, r) in contexts]
        token_ids, token_mask = nelutils.make_equal_len(token_ids, word_voca.unk_id)
        token_ids_t = torch.LongTensor(token_ids)
        token_mask_t = torch.FloatTensor(token_mask)

        # entity_ids is the cand ids for each mention in batch
        wiki_prefix = 'en.wikipedia.org/wiki/'
        ent_candss = []
        for m in mentions:
            named_cands = [c.text for c in m.candidates]
            cands = [(wiki_prefix + c).replace('"', '%22').replace(' ', '_') for c in
                     named_cands]  # They use the opposite direction mapping to me (add prefix, use URL strings)
            ent_candss.append(cands)
        #        if len(cands) == 0:
        #            print("Should remove this candidate")
        #            quit(0)

        entity_ids = [[
            entity_voca.get_id(ent_cand) for ent_cand in ent_cands
        ]
            for ent_cands in ent_candss]

        entity_ids, entity_mask = nelutils.make_equal_len(entity_ids, entity_voca.unk_id)
        entity_ids = torch.LongTensor(entity_ids)
        entity_mask = torch.FloatTensor(entity_mask)
        # scores (n_ment*8)
        scores = model.forward(token_ids_t, token_mask_t, entity_ids, entity_mask, p_e_m=None)

        # print("their scores",scores)
        ######/THEIRS

        ######MINE
        mine = NeuralNet()
        mentions = doc.mentions
        n = len(mentions)
        embeddings, embed_mask = mine.embeddings(mentions, n)
        token_embeddingss, token_maskss = mine.token_embeddingss(mentions)
        psiss = mine.psiss(n, embeddings, token_embeddingss, token_maskss)
        # print("my scores",psiss)
        ######/MINE

        # diff = scores-psiss
        # print("scorediff",diff)
        max_error = utils.max_error(scores, psiss)
        print(scores, psiss)
        print(f"MaxError: {max_error}")
        self.assertTrue(max_error == 0)

    def test_fmc_consistency(self):
        # This'll be fun...
        # It appears they use a 2nd embedding (work out what this is from here!)
        # It appears they mean (not +), and add 1e5. Fun.
        # (Most of) this is in mulrel_ranker, I'm not even 100% sure what their vector dimensions are

        #       snd_word_voca, snd_word_embeddings = nelutils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
        #                                                                 voca_emb_dir + '/glove/word_embeddings.npy')
        SETTINGS.allow_nans = False
        # SETTINGS.dataset = Dataset()
        # SETTINGS.dataset.documents = [self.testing_doc2]
        doc = self.testing_doc3
        mentions = doc.mentions

        #####THEIRS (removed CUDA dependency)
        import sys
        sys.path.insert(1, "mulrel-nel-master/")
        import nel.local_ctx_att_ranker as local_ctx_att_ranker
        import nel.utils as nelutils

        # Load voca & make config
        voca_emb_dir = "/home/harrison/Documents/project/data/generated/embeddings/word_ent_embs/"  # SETTINGS.data_dir_embeddings
        word_voca, word_embeddings = nelutils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                             voca_emb_dir + 'word_embeddings.npy')
        entity_voca, entity_embeddings = nelutils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                                 voca_emb_dir + 'entity_embeddings.npy')
        config = {
            'snd_local_ctx_window': 6
        }
        print("CONFIG", config)
        model = local_ctx_att_ranker.LocalCtxAttRanker(config)
        # p_e_m is None (i think, from mulrel_ranker), although is a FloatTensor from ed_ranker, is never passed to local_ctx.
        # Compute tokens using their code

        for m in doc.mentions:
            snd_lctx = [self.model.snd_word_voca.get_id(t)
                        for t in sent[max(0, start - config['snd_local_ctx_window'] // 2):start]]
            snd_rctx = [self.model.snd_word_voca.get_id(t)
                        for t in sent[end:min(len(sent), end + config['snd_local_ctx_window'] // 2)]]
            snd_ment = [self.model.snd_word_voca.get_id(t)
                        for t in sent[start:end]]

            if len(snd_lctx) == 0:
                snd_lctx = [self.model.snd_word_voca.unk_id]
            if len(snd_rctx) == 0:
                snd_rctx = [self.model.snd_word_voca.unk_id]
            if len(snd_ment) == 0:
                snd_ment = [self.model.snd_word_voca.unk_id]

        contexts = []
        for m in doc.mentions:
            lctx = m.left_context.strip().split()
            lctx_ids = [word_voca.get_id(t) for t in lctx if nelutils.is_important_word(t)]
            lctx_ids = [tid for tid in lctx_ids if
                        tid != word_voca.unk_id]  # TODO - assuming self.prerank_model.word_voca is word_voca and not snd_voca (99% sure I'm right)
            lctx_ids = lctx_ids[max(0, len(lctx_ids) - config['ctx_window'] // 2):]

            rctx = m.right_context.strip().split()
            rctx_ids = [word_voca.get_id(t) for t in rctx if nelutils.is_important_word(t)]
            rctx_ids = [tid for tid in rctx_ids if tid != word_voca.unk_id]
            rctx_ids = rctx_ids[:min(len(rctx_ids), config['ctx_window'] // 2)]

            contexts.append((lctx_ids, rctx_ids))

        # token_ids is left@right context, or [unk], LEFT padded with unks, to equal length per mention, long tensor
        # token_mask is 1 iff exists for token_ids, including the initial [unk] if no ctx
        token_ids = [l + r
                     if len(l) + len(r) > 0
                     else [word_voca.unk_id]
                     for (l, r) in contexts]
        token_ids, token_mask = nelutils.make_equal_len(token_ids, word_voca.unk_id)
        token_ids_t = torch.LongTensor(token_ids)
        token_mask_t = torch.FloatTensor(token_mask)

        # entity_ids is the cand ids for each mention in batch
        wiki_prefix = 'en.wikipedia.org/wiki/'
        ent_candss = []
        for m in mentions:
            named_cands = [c.text for c in m.candidates]
            cands = [(wiki_prefix + c).replace('"', '%22').replace(' ', '_') for c in
                     named_cands]  # They use the opposite direction mapping to me (add prefix, use URL strings)
            ent_candss.append(cands)
        #        if len(cands) == 0:
        #            print("Should remove this candidate")
        #            quit(0)

        entity_ids = [[
            entity_voca.get_id(ent_cand) for ent_cand in ent_cands
        ]
            for ent_cands in ent_candss]

        entity_ids, entity_mask = nelutils.make_equal_len(entity_ids, entity_voca.unk_id)
        entity_ids = torch.LongTensor(entity_ids)
        entity_mask = torch.FloatTensor(entity_mask)
        # scores (n_ment*8)
        scores = model.forward(token_ids_t, token_mask_t, entity_ids, entity_mask, p_e_m=None)

        ######/THEIRS

        ######MINE
        mine = NeuralNet()
        # n = len(mentions)
        # embeddingss, embed_mask = mine.embeddingss(mentions, n)
        # token_embeddingss, tokenMaskss = mine.token_embeddingss(mentions)
        # psiss = mine.psiss(n, embeddingss, token_embeddingss, tokenMaskss)
        ######/MINE

        # diff = scores-psiss
        # print("scorediff",diff)
        # maxError = utils.maxError(scores, psiss)
        # print(f"MaxError: {maxError}")
        # self.assertTrue(maxError == 0)

        print("test ran without error")

    def test_load_train(self):
        import datasets
        dataset_train = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        print("loaded dataset successfully")

    def load_consistency(self, ident):
        path = f"/home/harrison/Documents/project/mount2/consistency/mrn_{ident}.pt"
        return torch.load(path, map_location=torch.device('cpu'))

    def test_psi_consistency_new(self):
        theirs = self.load_consistency("psi")

        net = NeuralNet()
        dataset = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        modeller.candidate_selection(dataset, "TEST_train", True)
        mentions = dataset.documents[0].mentions
        n = len(mentions)
        embeddings, embed_mask = net.embeddings(mentions, n)
        token_embeddingss, token_maskss = net.token_embeddingss(mentions)
        mine = net.psiss(n, embeddings, token_embeddingss, token_maskss)

        print("Their shape:", theirs.shape)
        print("My shape:", mine.shape)
        print(theirs, mine)
        close = torch.allclose(theirs, mine)
        self.assertTrue(close)

    def test_candidate_select_consistency_internal(self):
        pass

    def test_psi_consistency_internal(self):
        their_tok_ids = self.load_consistency("psi_i_tokid")
        their_tok_m = self.load_consistency("psi_i_tokm")
        their_ent_ids = self.load_consistency("psi_i_entid")
        their_ent_m = self.load_consistency("psi_i_entm")
        their_pem = self.load_consistency("psi_i_pem")
        their_att_mat = self.load_consistency("psi_i_attmat")
        their_tok_mat = self.load_consistency("psi_i_tokmat")
        their_psi_scores = self.load_consistency("psi_internal")

        # OURS
        SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        SETTINGS.dataset_eval = SETTINGS.dataset_train
        SETTINGS.normalisation = NormalisationMethod.MENT_NORM
        import our_consistency
        our_consistency.TESTING = True
        modeller.candidate_selection(SETTINGS.dataset_train, "CONSISTENCY", True)
        model = Model()
        model.neural_net = neural.NeuralNet()
        model.evals = EvalHistory()
        neural.train(model, lr=0)  # Just run 1 full training step
        our_tok_ids = our_consistency.SAVED["psi_i_tokid"]
        our_tok_m = our_consistency.SAVED["psi_i_tokm"]
        our_ent_ids = our_consistency.SAVED["psi_i_entid"]
        our_ent_m = our_consistency.SAVED["psi_i_entm"]
        our_pem = our_consistency.SAVED["psi_i_pem"]
        our_att_mat = our_consistency.SAVED["psi_i_attmat"]
        our_tok_mat = our_consistency.SAVED["psi_i_tokmat"]
        our_psi_scores = our_consistency.SAVED["psi_internal"]

        #####THEIRS (removed CUDA dependency)
        self.assertTrue(torch.equal(their_tok_ids, our_tok_ids))
        self.assertTrue(torch.equal(their_tok_m, our_tok_m))
        # sort()[0] gives sorted, default is on last dim
        their_sorted, their_sorting = torch.sort(their_ent_ids)  # ACBD [0,2,1,3] (sorted->their)
        # invert 8-element sorting by selecting into a [0,..,0],[7,..,7] matrix and nonzeroing their row indexes
        #        inv_their_sorting = (torch.arange(0,8).reshape(8,1).repeat(1,8)==their_sorting).nonzero()[:,1]
        # do this for 30 mentions
        inv_their_sorting = (torch.arange(0, 8).reshape(1, 8, 1).repeat(30, 1, 8) == their_sorting.reshape(30, 1,
                                                                                                           8)).nonzero().reshape(
            30, 8, 3)[:, :, 2]
        our_sorted, our_sorting = torch.sort(our_ent_ids)  # DCAB [2,3,1,0] (sorted->our)
        sorting = torch.gather(our_sorting, index=inv_their_sorting,
                               dim=1)  # their idx -> our idx # [2,1,3,0] (their->our) = (their->sorted)(sorted->our)
        self.assertTrue(torch.equal(their_sorted, our_sorted))
        self.assertTrue(torch.equal(their_ent_ids, torch.gather(our_ent_ids, index=sorting, dim=1)))
        self.assertTrue(torch.equal(their_ent_m, our_ent_m))
        self.assertTrue(their_pem == our_pem)
        self.assertTrue(torch.equal(their_att_mat, our_att_mat))
        self.assertTrue(torch.equal(their_tok_mat, our_tok_mat))
        self.assertTrue(torch.allclose(their_psi_scores, torch.gather(our_psi_scores, index=sorting, dim=1)))

    def test_pre_rank_internal(self):
        their_tok_ids = self.load_consistency("prerank_tokids")
        their_tok_offs = self.load_consistency("prerank_tokoffs")
        their_ent_ids = self.load_consistency("prerank_entids")
        their_ctx_idxs = self.load_consistency("top_pos")
        their_ctx_vals = self.load_consistency("top_pos_vals")
        their_ctx_scores = self.load_consistency("ntee_scores")
        their_ctx_ents = self.load_consistency("ntee_ents")
        their_ctx_sents = self.load_consistency("ntee_sents")

        #####MINE
        SETTINGS.dataset = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        SETTINGS.normalisation = NormalisationMethod.MENT_NORM
        import our_consistency
        our_consistency.TESTING = True
        modeller.candidate_selection(SETTINGS.dataset, "CONSISTENCY", True)
        our_tok_ids = our_consistency.SAVED["tokids"]
        our_tok_offs = our_consistency.SAVED["tokoffs"]
        our_ent_ids = our_consistency.SAVED["entids"]
        our_ctx_idxs = our_consistency.SAVED["top_pos"]
        our_ctx_vals = our_consistency.SAVED["top_pos_vals"]
        our_ctx_scores = our_consistency.SAVED["ntee_scores"]
        our_ctx_ents = our_consistency.SAVED["ntee_ents"]
        our_ctx_sents = our_consistency.SAVED["ntee_sents"]

        self.assertTrue(torch.allclose(their_tok_ids, our_tok_ids))
        self.assertTrue(torch.allclose(their_tok_offs, our_tok_offs))
        self.assertTrue(torch.allclose(their_ent_ids, our_ent_ids))
        # print("CAND 0,0",SETTINGS.dataset.documents[0].mentions[0].candidates[0].text)
        # print("THEIR EMBED",their_ctx_ents[0])
        # print("OUR EMBED",our_ctx_ents[0])
        # print("OUR ID",processeddata.ent_to_ent_id["Canadians of German ethnicity"])
        # print("OUR ID",our_ent_ids[0][0])
        torch.set_printoptions(precision=15)  # Print in high precision mode
        import numpy as np
        np.set_printoptions(precision=15)
        # print("THEIR EMBED",their_ctx_ents[3][15])
        # print("OUR EMBED",our_ctx_ents[3][15])
        # print(their_ctx_ents[3][15]-our_ctx_ents[3][15])
        self.assertTrue(torch.allclose(their_ctx_ents, our_ctx_ents))
        self.assertTrue(torch.allclose(their_ctx_sents, our_ctx_sents))
        # print("SCORESSHAPE",their_ctx_scores.shape,our_ctx_scores.shape)
        # print("SCORESDIFF", their_ctx_scores- our_ctx_scores)
        # print("SCORES",their_ctx_scores,our_ctx_scores)
        self.assertTrue(torch.allclose(their_ctx_scores, our_ctx_scores))
        import numpy as np

        if not np.array_equal(their_ctx_idxs, our_ctx_idxs):
            # Not exactly, equal, topk wasn't stable
            non_zero_idxs1, non_zero_idxs2 = np.nonzero(their_ctx_idxs - our_ctx_idxs)
            for non_zero_idx1, non_zero_idx2 in zip(non_zero_idxs1, non_zero_idxs2):
                their_ctx_idx = their_ctx_idxs[non_zero_idx1][non_zero_idx2]
                our_ctx_idx = our_ctx_idxs[non_zero_idx1][non_zero_idx2]
                their_ent_id = their_ent_ids[non_zero_idx1][their_ctx_idx]
                our_ent_id = our_ent_ids[non_zero_idx1][our_ctx_idx]
                self.assertTrue(their_ent_id == our_ent_id)
        # self.assertTrue(torch.allclose(their_ctx_vals,our_ctx_vals))#don't check because we don't bother normalising these

    def test_embedding_consistency(self):
        their_sent = self.load_consistency("embs_i_sent")  # from ed_ranker
        their_lctx = self.load_consistency("embs_i_lctx")
        # OURS
        SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        SETTINGS.dataset_eval = SETTINGS.dataset_train
        SETTINGS.normalisation = NormalisationMethod.MENT_NORM
        import our_consistency
        our_consistency.TESTING = True
        modeller.candidate_selection(SETTINGS.dataset_train, "CONSISTENCY", True)
        model = Model()
        model.neural_net = neural.NeuralNet()
        model.evals = EvalHistory()
        neural.train(model, lr=0)  # Just run 1 full training step
        our_sent = our_consistency.SAVED["embs_i_sent"]
        our_l_ctx = our_consistency.SAVED["embs_i_lctx"]
        # '1000' is the only token with 2 second embeddingss
        # print(processeddata.word_to_word_id_snd["1000"],566,577)
        # import numpy as np
        # self.assertTrue(np.array_equal(processeddata.word_id_to_embedding_snd[566],processeddata.word_id_to_embedding_snd[577]))

        self.assertTrue(their_sent == our_sent)
        self.assertTrue(their_lctx == our_l_ctx)

    def test_fmc_consistency(self):
        their_ctx_bow = self.load_consistency("bow_ctx_vecs")
        their_l_ctx_embs = self.load_consistency("fmc_i_lctx_embs")
        their_m_ctx_embs = self.load_consistency("fmc_i_mctx_embs")
        their_r_ctx_embs = self.load_consistency("fmc_i_rctx_embs")
        their_l_ctx_score = self.load_consistency("fmc_i_lctx_score")
        their_m_ctx_score = self.load_consistency("fmc_i_mctx_score")
        their_r_ctx_score = self.load_consistency("fmc_i_rctx_score")
        their_l_ctx_ids = self.load_consistency("fmc_i_lctx_ids")
        their_m_ctx_ids = self.load_consistency("fmc_i_mctx_ids")
        their_r_ctx_ids = self.load_consistency("fmc_i_rctx_ids")
        their_initial_weight = self.load_consistency("fmc_initialweight")
        their_pre_weight = self.load_consistency("fmc_preweight")
        their_input = self.load_consistency("fmc_input")
        their_model = self.load_consistency("fmc_model")
        their_output = self.load_consistency("fmc_output")

        # OURS
        SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        SETTINGS.dataset_eval = SETTINGS.dataset_train
        SETTINGS.normalisation = NormalisationMethod.MENT_NORM
        import our_consistency
        our_consistency.TESTING = True
        modeller.candidate_selection(SETTINGS.dataset_train, "CONSISTENCY", True)
        model = Model()
        model.neural_net = neural.NeuralNet()
        model.evals = EvalHistory()
        neural.train(model, lr=0)  # Just run 1 full training step
        our_ctx_bow = our_consistency.SAVED["bow_ctx_vecs"]
        our_l_ctx_embs = our_consistency.SAVED["fmc_i_lctx_embs"]
        our_m_ctx_embs = our_consistency.SAVED["fmc_i_mctx_embs"]
        our_r_ctx_embs = our_consistency.SAVED["fmc_i_rctx_embs"]
        our_l_ctx_score = our_consistency.SAVED["fmc_i_lctx_score"]
        our_m_ctx_score = our_consistency.SAVED["fmc_i_mctx_score"]
        our_r_ctx_score = our_consistency.SAVED["fmc_i_rctx_score"]
        our_l_ctx_ids = our_consistency.SAVED["fmc_i_lctx_ids"]
        our_m_ctx_ids = our_consistency.SAVED["fmc_i_mctx_ids"]
        our_r_ctx_ids = our_consistency.SAVED["fmc_i_rctx_ids"]
        our_initial_weight = our_consistency.SAVED["fmc_initialweight"]
        our_pre_weight = our_consistency.SAVED["fmc_preweight"]
        our_input = our_consistency.SAVED["fmc_input"]
        our_model = our_consistency.SAVED["fmc_model"]
        our_output = our_consistency.SAVED["fmc_output"]

        self.assertTrue(torch.equal(their_l_ctx_ids, our_l_ctx_ids))
        self.assertTrue(torch.equal(their_m_ctx_ids, our_m_ctx_ids))
        self.assertTrue(torch.equal(their_r_ctx_ids, our_r_ctx_ids))

        self.assertTrue(torch.equal(their_l_ctx_embs, our_l_ctx_embs))
        self.assertTrue(torch.equal(their_m_ctx_embs, our_m_ctx_embs))
        self.assertTrue(torch.equal(their_r_ctx_embs, our_r_ctx_embs))

        self.assertTrue(torch.equal(their_l_ctx_score, our_l_ctx_score))
        self.assertTrue(torch.equal(their_m_ctx_score, our_m_ctx_score))
        self.assertTrue(torch.equal(their_r_ctx_score, our_r_ctx_score))

        self.assertTrue(torch.equal(their_ctx_bow, our_ctx_bow))
        self.assertTrue(torch.equal(their_input, our_input))
        for their_param, our_param in zip(their_initial_weight, our_initial_weight):
            self.assertTrue(torch.equal(their_param, our_param))
        for their_param, our_param in zip(their_pre_weight, our_pre_weight):
            self.assertTrue(torch.equal(their_param, our_param))
        self.assertTrue(str(their_model) == str(our_model))
        self.assertTrue(torch.equal(their_output, our_output))

    def test_phi_consistency(self):
        # assuming fmc consistency
        their_pad_ent = self.load_consistency("use_pad_ent")
        their_mode = self.load_consistency("mode")
        their_phi = self.load_consistency("phi")
        their_comp_mode = self.load_consistency("comp_mode")
        their_phi_k = self.load_consistency("phi_k")
        their_phi_rel_ent = self.load_consistency("phi_i_relent")
        their_phi_ent = self.load_consistency("phi_i_ent")
        their_phi_rel = self.load_consistency("phi_i_rel")
        their_ctx = self.load_consistency("ctx_vecs")
        their_exp_mm = self.load_consistency("exp_i_mentment")
        their_exp_mm_1 = self.load_consistency("exp_i_mentment_1")
        their_exp_mm_2 = self.load_consistency("exp_i_mentment_2")
        their_exp_mm_s = self.load_consistency("exp_i_mentment_scaled")
        their_rel_ctx_ctx = self.load_consistency("rel_ctx_ctx")
        their_exp_mm_prob = self.load_consistency("exp_i_mentment_probs")

        # OURS
        SETTINGS.dataset_train = datasets.load_dataset("aida_train.csv", "AIDA/aida_train.txt")
        SETTINGS.dataset_eval = SETTINGS.dataset_train
        SETTINGS.normalisation = NormalisationMethod.MENT_NORM
        import our_consistency
        our_consistency.TESTING = True
        modeller.candidate_selection(SETTINGS.dataset_train, "CONSISTENCY", True)
        model = Model()
        model.neural_net = neural.NeuralNet()
        model.evals = EvalHistory()
        neural.train(model, lr=0)  # Just run 1 full training step
        our_pad_ent = our_consistency.SAVED["use_pad_ent"]
        our_mode = our_consistency.SAVED["mode"]
        our_phi = our_consistency.SAVED["phi"]
        our_comp_mode = our_consistency.SAVED["comp_mode"]
        our_phi_k = our_consistency.SAVED["phi_k"]
        our_phi_rel_ent = our_consistency.SAVED["phi_i_relent"]
        our_phi_ent = our_consistency.SAVED["phi_i_ent"]
        our_phi_rel = our_consistency.SAVED["phi_i_rel"]
        our_ctx = our_consistency.SAVED["ctx_vecs"]
        our_exp_mm = our_consistency.SAVED["exp_i_mentment"]
        our_exp_mm_1 = our_consistency.SAVED["exp_i_mentment_1"]
        our_exp_mm_2 = our_consistency.SAVED["exp_i_mentment_2"]
        our_exp_mm_s = our_consistency.SAVED["exp_i_mentment_scaled"]
        our_rel_ctx_ctx = our_consistency.SAVED["rel_ctx_ctx"]  # TODO this will be different if >1000 mentions
        our_exp_mm_prob = our_consistency.SAVED["exp_i_mentment_probs"]

        self.assertTrue(their_pad_ent == our_pad_ent)
        self.assertTrue(their_mode == our_mode)
        self.assertTrue(their_comp_mode == our_comp_mode)
        # TODO - check - is the padding vector normalised in the same way as as the actual embeddingss?
        self.assertTrue(torch.equal(their_ctx, our_ctx))

        # exp brackets
        self.assertTrue(torch.allclose(our_exp_mm, their_exp_mm))
        self.assertTrue(torch.allclose(our_exp_mm_1, their_exp_mm_1))
        self.assertTrue(torch.allclose(our_exp_mm_2, their_exp_mm_2))
        self.assertTrue(torch.allclose(our_exp_mm_s, their_exp_mm_s))

        print(their_rel_ctx_ctx.shape, our_rel_ctx_ctx.shape)
        print(their_rel_ctx_ctx, our_rel_ctx_ctx)
        self.assertTrue(torch.allclose(their_rel_ctx_ctx, our_rel_ctx_ctx))

        print(our_exp_mm_prob, their_exp_mm_prob)
        self.assertTrue(torch.allclose(our_exp_mm_prob, their_exp_mm_prob))

        print(their_phi_ent.shape, our_phi_ent.shape)
        self.assertTrue(torch.allclose(their_phi_ent, our_phi_ent.reshape(31, 8, 300)))
        print(their_phi_rel.shape, our_phi_rel.shape)
        self.assertTrue(torch.allclose(their_phi_rel, our_phi_rel))
        print(their_phi_rel_ent.shape, our_phi_rel_ent.shape)
        self.assertTrue(torch.allclose(their_phi_rel_ent, our_phi_rel_ent.transpose(0, 1)))
        print(their_phi_k.shape, our_phi_k.shape)
        self.assertTrue(torch.allclose(their_phi_k, our_phi_k.transpose(0, 4)))
        print(their_phi.shape, our_phi.shape)
        self.assertTrue(torch.equal(their_phi, our_phi.transpose(1, 2)))

# REMAINING CONSISTENCY:
#
# rel_ctx_ctx
