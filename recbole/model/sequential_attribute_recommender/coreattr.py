# -*- coding: utf-8 -*-

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de

r"""
CORE
################################################
Reference:
    Yupeng Hou, Binbin Hu, Zhiqiang Zhang, Wayne Xin Zhao. "CORE: Simple and Effective Session-based Recommendation
    within Consistent Representation Space." in SIGIR 2022.

    https://github.com/RUCAIBox/CORE
"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, \
    embed_attributes, merge_embedded_item_features, create_mask_or_pad_dict, concat_user_embeddings, \
    merge_user_embeddings, embed_user_attributes
from recbole.model.sequential_recommender.core import CORE

class COREAttr(CORE):
    r"""CORE is a simple and effective framewor, which unifies the representation space
    for both the encoding and decoding processes in session-based recommendation.
    """

    def __init__(self, config, dataset):
        super(COREAttr, self).__init__(config, dataset)
        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.embedding_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.embedding_size)
        self.user_fusion = None if self.user_attributes is None else self.user_attributes.get("user_fusion", "concat")
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        if self.user_fusion == "concat":
            self.final_seq_length = self.max_seq_length + 1
            self.net.position_embedding = nn.Embedding(
                self.final_seq_length, config["embedding_size"])

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_emb = self.item_embedding(item_seq)
        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=False, pad_values=self.pad_dict)
        item_seq_emb = merge_embedded_item_features(embedded_features, self.item_attributes, item_seq_emb)

        embedded_user_features = embed_user_attributes(interaction, self.user_attributes, self.user_attribute_embeddings)
        item_seq_mask = item_seq
        if self.user_fusion == "pre_merge":
            item_seq_emb = merge_user_embeddings(self.user_attributes, embedded_user_features, sequence=item_seq_emb)
        if self.user_fusion == "concat":
            item_seq_emb = concat_user_embeddings(self.user_attributes, embedded_user_features, item_seq_emb)
            user_mask = torch.ones((item_seq.size(0), 1), device=item_seq.device, dtype=item_seq.dtype)
            item_seq_mask = torch.concat((user_mask, item_seq), dim=1)

        x = self.sess_dropout(item_seq_emb)
        # Representation-Consistent Encoder (RCE)
        alpha = self.net(item_seq_mask, x)
        seq_output = torch.sum(alpha * x, dim=1)
        if self.user_fusion == "post_merge":
            seq_output = merge_user_embeddings(self.user_attributes, embedded_user_features, state=seq_output)
        seq_output = F.normalize(seq_output, dim=-1)
        return seq_output

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        all_item_emb = self.item_embedding.weight
        all_item_emb = self.item_dropout(all_item_emb)
        # Robust Distance Measuring (RDM)
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = (
                torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        )
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) / self.temperature
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding.weight
        # no dropout for evaluation
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = (
                torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        )
        return scores
