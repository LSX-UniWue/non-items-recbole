# -*- coding: utf-8 -*-
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de
r"""
NextItNet
################################################

Reference:
    Fajie Yuan et al., "A Simple Convolutional Generative Network for Next Item Recommendation" in WSDM 2019.

Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import RegLoss, BPRLoss
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, \
    embed_attributes, merge_embedded_item_features, concat_user_embeddings, create_mask_or_pad_dict
from recbole.model.sequential_recommender import NextItNet


class NextItNetAttr(NextItNet):
    r"""The network architecture of the NextItNet model is formed of a stack of holed convolutional layers, which can
    efficiently increase the receptive fields without relying on the pooling operation.
    Also residual block structure is used to ease the optimization for much deeper networks.

    Note:
        As paper said, for comparison purpose, we only predict the next one item in our evaluation,
        and then stop the generating process. Although the number of parameters in residual block (a) is less
        than it in residual block (b), the performance of b is better than a.
        So in our model, we use residual block (b).
        In addition, when dilations is not equal to 1, the training may be slow. To  speed up the efficiency, please set the parameters "reproducibility" False.
    """

    def __init__(self, config, dataset):
        super(NextItNetAttr, self).__init__(config, dataset)

        self.item_attributes = config["items"]
        self.user_attributes = config["users"]

        # define layers and loss
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.embedding_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.embedding_size)
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        self.apply(self._init_weights)

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_emb = self.item_embedding(item_seq)

        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=False, pad_values=self.pad_dict)
        item_seq_emb = merge_embedded_item_features(embedded_features, self.item_attributes, item_seq_emb)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.attribute_embeddings)
        item_seq_emb = concat_user_embeddings(self.user_attributes,embedded_user_features, item_seq_emb)

        # [batch_size, seq_len, embed_size]
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        hidden = dilate_outputs[:, -1, :].view(
            -1, self.residual_channels
        )  # [batch_size, embed_size]
        seq_output = self.final_layer(hidden)  # [batch_size, embedding_size]
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
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        reg_loss = self.reg_loss([self.item_embedding.weight, self.final_layer.weight])
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_rb()
        return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, item_num]
        return scores
