# -*- coding: utf-8 -*-
# @Time   : 2020/8/25 19:56
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE
# @Time   : 2020/9/15, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de

r"""
NARM with Attribute Embeddings
################################################

Reference:
    Jing Li et al. "Neural Attentive Session-based Recommendation." in CIKM 2017.

Reference code:
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, \
    embed_attributes, merge_embedded_item_features, concat_user_embeddings, create_mask_or_pad_dict
from recbole.model.sequential_recommender import NARM


class NARMAttr(SequentialRecommender):
    r"""NARM explores a hybrid encoder with an attention mechanism to model the user’s sequential behavior,
    and capture the user’s main purpose in the current session.

    """

    def __init__(self, config, dataset):
        super(NARMAttr, self).__init__(config, dataset)

        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.n_layers = config["n_layers"]
        self.dropout_probs = config["dropout_probs"]
        self.device = config["device"]
        self.init_type = "recbole_init" if not hasattr(config,"init_type") else config["init_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.embedding_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.embedding_size)
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        self.emb_dropout = nn.Dropout(self.dropout_probs[0])
        self.gru = nn.GRU(
            self.embedding_size,
            self.hidden_size,
            self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_probs[1])
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)
        self.loss_type = config["loss_type"]
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        if self.init_type == "recbole_init":
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)



    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_seq_emb = self.item_embedding(item_seq)
        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=False, pad_values=self.pad_dict)
        item_seq_emb = merge_embedded_item_features(embedded_features, self.item_attributes, item_seq_emb)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.attribute_embeddings)
        item_seq_emb = concat_user_embeddings(self.user_attributes,embedded_user_features, item_seq_emb)
        if self.user_attributes is not None:
            item_seq_len = item_seq_len + 1


        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        if self.user_attributes is not None:
            user_mask = torch.ones(item_seq.size(0), 1, dtype=torch.int64, device=item_seq.device)
            mask_seq = torch.cat((user_mask, item_seq), dim=1)
        else:
            mask_seq = item_seq

        mask = mask_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        return seq_output

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
