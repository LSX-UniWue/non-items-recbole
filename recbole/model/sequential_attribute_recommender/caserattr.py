# -*- coding: utf-8 -*-
# @Time   : 2020/9/21
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/2
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de

r"""
Caser with  Attribute Embeddings
################################################

Reference:
    Jiaxi Tang et al., "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding" in WSDM 2018.

Reference code:
    https://github.com/graytowne/caser_pytorch

"""

import torch
from torch import nn
from torch.nn import functional as F

from recbole.model.loss import RegLoss, BPRLoss
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, \
    embed_attributes, merge_embedded_item_features, concat_user_embeddings, create_mask_or_pad_dict
from recbole.model.sequential_recommender import Caser


class CaserAttr(Caser):
    r"""Caser is a model that incorporate CNN for recommendation.

    Note:
        We did not use the sliding window to generate training instances as in the paper, in order that
        the generation method we used is common to other sequential models.
        For comparison with other models, we set the parameter T in the paper as 1.
        In addition, to prevent excessive CNN layers (ValueError: Training loss is nan), please make sure
        the parameters MAX_ITEM_LIST_LENGTH small, such as 10.
    """

    def __init__(self, config, dataset):
        super(Caser, self).__init__(config, dataset)

        # load parameters info
        self.L = config["L"]
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]
        self.n_h = config["nh"]
        self.n_v = config["nv"]
        self.dropout_prob = config["dropout_prob"]
        self.reg_weight = config["reg_weight"]

        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.vertical_kernel_size = self.max_seq_length

        # define layers and loss
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.embedding_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.embedding_size)
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        if self.user_attributes is not None:
            self.vertical_kernel_size = self.max_seq_length + 1

        # load dataset info
        self.n_users = dataset.user_num

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.vertical_kernel_size, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size), ) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()
        self.reg_loss = RegLoss()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]

        # Embedding Look-up
        item_seq_emb = self.item_embedding(item_seq)
        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=False, pad_values=self.pad_dict)
        item_seq_emb = merge_embedded_item_features(embedded_features, self.item_attributes, item_seq_emb)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.attribute_embeddings)
        item_seq_emb = concat_user_embeddings(self.user_attributes, embedded_user_features, item_seq_emb)

        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq_emb = item_seq_emb.unsqueeze(1)
        user_emb = self.user_embedding(user).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x))
        # the hidden_state of the predicted item, size: (batch_size * hidden_size)
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

        reg_loss = self.reg_loss(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
                self.conv_v.weight,
                self.fc1.weight,
                self.fc2.weight,
            ]
        )
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_conv_h()
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
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
