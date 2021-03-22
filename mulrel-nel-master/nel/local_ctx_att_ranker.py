# ORIGINAL FILE FROM https://github.com/lephong/mulrel-nel/
# THIS VERSION MAY CONTAIN MODIFICATIONS
import nel.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from nel.abstract_word_entity import AbstractWordEntity


class LocalCtxAttRanker(AbstractWordEntity):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker, self).__init__(config)

        self.hid_dims = config['hid_dims']
        self.tok_top_n = config['tok_top_n']
        self.margin = config['margin']

        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.local_ctx_dr = nn.Dropout(p=0)

        self.score_combine_linear_1 = nn.Linear(2, self.hid_dims)
        self.score_combine_act_1 = nn.ReLU()
        self.score_combine_linear_2 = nn.Linear(self.hid_dims, 1)

    def print_weight_norm(self):
        print('att_mat_diag', self.att_mat_diag.data.norm())
        print('tok_score_mat_diag', self.tok_score_mat_diag.data.norm())
        print('f - l1.w, b', self.score_combine_linear_1.weight.data.norm(),
              self.score_combine_linear_1.bias.data.norm())
        print('f - l2.w, b', self.score_combine_linear_2.weight.data.norm(),
              self.score_combine_linear_2.bias.data.norm())

    def print_attention(self, gold_pos):
        token_ids = self._token_ids.data.cpu().numpy()
        entity_ids = self._entity_ids.data.cpu().numpy()
        att_probs = self._att_probs.data.cpu().numpy()
        top_tok_att_ids = self._top_tok_att_ids.data.cpu().numpy()
        gold_pos = gold_pos.data.cpu().numpy()
        scores = self._scores.data.cpu().numpy()

        print('===========================================')
        for tids, eids, ap, aids, gpos, ss in zip(token_ids, entity_ids, att_probs, top_tok_att_ids, gold_pos, scores):
            selected_tids = tids[aids]
            print('-------------------------------')
            print(utils.tokgreen(repr([(self.entity_voca.id2word[e], s) for e, s in zip(eids, ss)])),
                  utils.tokblue(repr(self.entity_voca.id2word[eids[gpos]] if gpos > -1 else 'UNKNOWN')))
            print([(self.word_voca.id2word[t], a[0]) for t, a in zip(selected_tids, ap)])

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m=None):
        #H NOTES:
        # Some numbers change between runs (which occur in a random order), some are constant
        # 8 is the number of candidate entities per mention, CONSTANT
        # n_ment is the number of mentions, the "batchsize", varies between runs
        # n_words is the number of words (per batch??) (I think this is the context window - local context window is 100 by default)
        #   context window is context around which passes the is_important_word check, algorithm as follows: (ed_ranker:195)
        #       strip and split left and right ctxs
        #       from each, filter out unimportant words, or those where id is unk
        #       take up to 50 from each side
        #       join the two sides (ed_ranker:288)
        #       pad on the right with unk id (ed_ranker:301) until all are the same length
        # 300 is the dimensions of an embedding, CONSTANT
        # 25 is the window size, CONSTANT
        #Harrison called from mulrel_ranker (using super) IFF use_local is True there (should be??)
        #Harrison entity_ids is therefore a (n_ments * n_cands) 2D tensor (note n_cands=8 typically)
        token_ids#2D (n_ment*n_words) tensor of IDs, 1st ID seems to be anything, last few ids are 492407 (#UNK#)
            # up to half ctx window either side (ed_ranker:198), per-mention
        tok_mask#2D (n_ment*n_words) binary tensor, 1 for usual IDs, 0 for unk ids it appears
        entity_ids#2D (n_ment*8) tensor of IDs (274474 is UNK)
        entity_mask#2D (n_ment*8) binary tensor mask, 1 for IDS, 0 for unk ids
        p_e_m#None

        batchsize, n_words = token_ids.size()#H n_ment*n_words
        n_entities = entity_ids.size(1) #Harrison =8 (n_cands) typically
        tok_mask = tok_mask.view(batchsize, 1, -1)
        tok_mask#Now (n_ment*1*n_words) (i.e. candidate doesn't matter)

        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.entity_embeddings(entity_ids)
        tok_vecs#H n_ment*n_words*300
        entity_vecs.size()#H n_ment*8*300

        # att
        #H                            n_ment*8*300 *      300         ,  n_ment*300*n_words
        ent_tok_att_scores = torch.bmm(entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores# (n_ment*8*n_words) which for each mention, candidate, and word, is score cand_i <dot> word_j weighted by att_mat_diag.
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(1e10))
        ent_tok_att_scores# (n_ment*8*n_words) as before, but set to -1e10 for unknowns
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        tok_att_scores# (n_ment*n_words) is highest score of some cand <dot> word weighted by att_mat_diag
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        top_tok_att_scores# (n_ment*25) is the 25 highest scores for some cand <dot> some word, weighted by att_mat_diag (typically around 0.1-0.3)
        top_tok_att_ids# (n_ment*25) is the ids of the above scores into tok_att_scores (0-n_words) (NOTE: not sorted best-to-worst)


        att_probs = F.softmax(top_tok_att_scores, dim=1)
        att_probs# (n_ment,25) softmax of the 25 highest scoring words
        att_probs = att_probs.view(batchsize, -1, 1)
        att_probs# (n_ment*25*1) as above
        att_probs_BK = att_probs.clone()
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)
        att_probs# (n_ment*25*1) as above, should be identical as sum should be 1 after softmax

        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        selected_tok_vecs# (n_ment*25*300) tensor of token vectors (the corresponding vectors for top_tok_att_ids)

        #NOTES:
        # top_tok_att_ids.view is a (n_ment*25*1) tensor of the top 25 word ids per mention for the anycand<dot>word weighted score
        # top_tok.view.repeat is a (n_ment*25*300) tensor of the ids as above, with the id repeated 300 times
        # tok_vecs is a (n_ment*n_words*300) tensor
        # gather gets a (n_ment*25*300) tensor where for each mention the id of the word is one of the ids in top_tok, this is repeated to get the right 300 elements.


        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        global DEBUG
        DEBUG['tes'] = ctx_vecs
#        a1 = torch.sum(selected_tok_vecs * att_probs, dim=1) * self.tok_score_mat_diag.repeat(batchsize,1) NO
#        a1 = torch.sum((selected_tok_vecs * att_probs) * self.tok_score_mat_diag, dim=1, keepdim=True) YES
#        a1 = torch.sum(selected_tok_vecs * att_probs, dim=1, keepdim=True) * self.tok_score_mat_diag YES
#        a1 = torch.sum(selected_tok_vecs * att_probs, dim=1) * self.tok_score_mat_diag
#        a1 = torch.matmul(torch.sum(selected_tok_vecs * att_probs, dim=1),self.tok_score_mat_diag.diag_embed())
        a1 = torch.matmul(torch.sum(selected_tok_vecs * att_probs, dim=1),self.tok_score_mat_diag.diag_embed()).view(batchsize,300,1).transpose(1,2)
        a2 = ctx_vecs#.view(batchsize,300)
        print("DIFFA?",a1-a2)

        ctx_vecs #(n_ment*1*300) tensor of ctx scores
        #tok_score_mat_diag is a 300 tensor of weights, so eqn ((seltok*diag)*attprob) is (n_ment*25*300) score, where it is the <dot> score of the top 25 words,
        # multiplied by a weighting (based on embedding position), and then multiplied by the softmax weighting of that word
        # ctx_vecs is then the sum of these weighted embedding scores

        ctx_vecs_BK = ctx_vecs.clone()#H backing up
        ctx_vecs = self.local_ctx_dr(ctx_vecs)#TODO Harrison note - this is a dropout with p=0 therefore identity???
        print("H: CV",ctx_vecs_BK.equal(ctx_vecs))#H True

        #                        n_ment*8*300 , n_ment*300*1 -> n_ment*8*1 -> n_ment*8
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(batchsize, n_entities)
        ent_ctx_scores# (n_ment*8) tensor of <dot> between each candidate entity and it's mentions context score
        #i.e. this is a sense of relatedness between the entity and the context around the mention
        #this is the e_i^T part of the psi calculation, where ctx_vecs is BF(c_i)

        # combine with p(e|m) if p_e_m is not None
        if p_e_m is not None:
            inputs = torch.cat([ent_ctx_scores.view(batchsize * n_entities, -1),
                                torch.log(p_e_m + 1e-20).view(batchsize * n_entities, -1)], dim=1)
            hidden = self.score_combine_linear_1(inputs)
            hidden = self.score_combine_act_1(hidden)
            scores = self.score_combine_linear_2(hidden).view(batchsize, n_entities)
        else:
            scores = ent_ctx_scores

        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))
        scores #(n_ment*8) tensor as before, set to -1e10 for unks

        # printing attention (debugging)
        self._token_ids = token_ids
        self._entity_ids = entity_ids
        self._att_probs = att_probs
        self._top_tok_att_ids = top_tok_att_ids
        self._scores = scores

        self._entity_vecs = entity_vecs
        self._local_ctx_vecs = ctx_vecs

        #Harrison from the use in mulrel_ranker it is clear that scores is meant to be a tensor representing local enttiy scores (psi)
        return scores
        #paper states this is e * B * f(c)
        #I determine this is:
        # e^T . CTX
        #CTX = SUM(25 'i's for best attscore_i) (seltokvec_i*DIAG*attprob_i)
        #  ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        #  selected_tok_vecs is (n_ment,25,300), embeddings for 25 selected tokens for each mention
        #  tok_score_mat_diag is (300) tensor, initially 1s
        #  attprob_i is just a numerical probability
        #  e . SUM(SEL*DIAG1*prob,1) = e . DIAG1*SUM(SEL*prob,1) (for a 300 e, 300 diag, sum 1 (multiplication is transpositional) to a 300
        #seltokvec_i = vector for token i (t_i)
        #attprob_i = softmax(attscore_i)
        #attscore_i = max(all entities j) (e_j <dot> t_i weighted by DIAG2)



    def regularize(self, max_norm=1):
        l1_w_norm = self.score_combine_linear_1.weight.norm()
        l1_b_norm = self.score_combine_linear_1.bias.norm()
        l2_w_norm = self.score_combine_linear_2.weight.norm()
        l2_b_norm = self.score_combine_linear_2.bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine_linear_1.weight.data = self.score_combine_linear_1.weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine_linear_1.bias.data = self.score_combine_linear_1.bias.data * max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine_linear_2.weight.data = self.score_combine_linear_2.weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine_linear_2.bias.data = self.score_combine_linear_2.bias.data * max_norm / l2_b_norm.data

    def loss(self, scores, true_pos):
        loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        return loss
