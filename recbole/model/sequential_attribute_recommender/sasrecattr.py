# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de

"""
SASRec with Attribute Embeddings
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
import torch
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, embed_attributes, \
    merge_embedded_item_features, concat_user_embeddings, create_mask_or_pad_dict
from recbole.model.sequential_recommender import SASRec

class SASRecAttr(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecAttr, self).__init__(config, dataset)

        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.hidden_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.hidden_size)
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        self.apply(self._init_weights)

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=False, pad_values=self.pad_dict)
        item_seq_emb = merge_embedded_item_features(embedded_features, self.item_attributes, input_emb)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.attribute_embeddings)
        item_seq_emb = concat_user_embeddings(self.user_attributes,embedded_user_features, item_seq_emb)
        if self.user_attributes is not None:
            item_seq_len = item_seq_len + 1
            user_mask = torch.ones(item_seq.size(0), 1, dtype=torch.int64, device=item_seq.device)
            mask_seq = torch.cat((user_mask, item_seq), dim=1)
        else:
            mask_seq = item_seq

        input_emb = self.LayerNorm(item_seq_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(mask_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

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
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(interaction)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
