# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

# UPDATE
# @Time   : 2023/9/4
# @Author : Enze Liu
# @Email  : enzeeliu@foxmail.com

# EXTENSION: Attribute Embeddings
# @Time   : 2024/9/1
# @Author  : Elisabeth Fischer
# @Email   : elisabeth.fischer@informatik.uni-wuerzburg.de

r"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    In CIKM 2019.

Reference code:
    The authors' tensorflow implementation https://github.com/FeiSun/BERT4Rec

"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, \
    embed_attributes, merge_embedded_item_features, create_mask_or_pad_dict, concat_user_embeddings

class BERT4RecAttr(SequentialRecommender):
    """
    BERT4RecAttr: based on BERT4Rec, we add attribute information to the model. Implementation differs from
    Recbole's version in some bugfixes to prevent NaNs in the loss. MaskedTraining is also adjusted (data.transform.py)
    """
    def __init__(self, config, dataset):
        super(BERT4RecAttr, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]
        self.init_type = "recbole_init" if not hasattr(config,"init_type") else config["init_type"]
        self.use_masked_features = True if not hasattr(config,"use_masked_features") else config["use_masked_features"]
        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in [ 'CE']!")

        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.hidden_size, masking=True)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.hidden_size)
        self.mask_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="mask")
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")

        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]


        # parameters initialization
        if self.init_type == "recbole_init":
            self.apply(self._init_weights)
        elif self.init_type == "xavier":
            self.apply(self._init_weights_xavier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_weights_xavier(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def reconstruct_test_data(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        interaction[
            self.MASK_ITEM_SEQ] = item_seq.clone()  # We don't want to use the masked sequence an neither want to change the original sequence
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        feature_tensor_dict = {}
        if self.item_attributes is not None:
            if self.item_attributes["attributes"] is not None:
                for (feature_name, infos) in self.item_attributes["attributes"].items():
                    feature_tensor_dict[feature_name] = interaction[feature_name + "_list"].clone()
            if self.item_attributes.get("item_id_type_settings", None) is not None:
                feature_tensor_dict[self.item_attributes["item_id_type_settings"]["name"]] = interaction[
                    self.item_attributes["item_id_type_settings"]["name"] + "_list"].clone()
        max_seq_len = self.max_seq_length
        for sample_id, seq_len in enumerate(item_seq_len):
            if max_seq_len == seq_len:
                interaction[self.MASK_ITEM_SEQ][sample_id,] = torch.concat([item_seq[sample_id,],
                                                                            torch.tensor([self.mask_token],
                                                                                         dtype=item_seq.dtype,
                                                                                         device=item_seq.device)])[1:]
                for (feature_name, feat_tensor) in feature_tensor_dict.items():
                    mask = self.mask_dict[feature_name]
                    feature_tensor_dict[feature_name][sample_id,] = torch.concat(
                        [feat_tensor[sample_id,],
                         torch.tensor([mask], dtype=feat_tensor.dtype, device=feat_tensor.device)])[1:]
            else:
                interaction[self.MASK_ITEM_SEQ][sample_id, seq_len] = self.mask_token
                for (feature_name, feat_tensor) in feature_tensor_dict.items():
                    mask = self.mask_dict[feature_name]
                    feature_tensor_dict[feature_name][sample_id, seq_len] = torch.tensor(mask,
                                                                                         dtype=feat_tensor.dtype,
                                                                                         device=feat_tensor.device)
                interaction[self.ITEM_SEQ_LEN][sample_id] += 1
                
        for (feature_name, feat_tensor) in feature_tensor_dict.items():
            interaction["mask_" + feature_name + "_list"] = feat_tensor
        return interaction

    def forward(self, interaction):
        item_seq = interaction[self.MASK_ITEM_SEQ]
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)

        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             use_masked_sequence=self.use_masked_features, pad_values=self.pad_dict)


        item_emb = merge_embedded_item_features(embedded_features, self.item_attributes, item_emb)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.attribute_embeddings)
        item_emb = concat_user_embeddings(self.user_attributes,embedded_user_features, item_emb)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output  # [B L H]


    def multi_hot_embed(self, masked_index, pos_items, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        pos_items = pos_items.view(-1)
        # If the pos item is the padding token, set to zero
        padding_tokens = pos_items == 0

        multi_hot[padding_tokens] = 0
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        masked_index = interaction[self.MASK_INDEX]

        seq_output = self.forward(interaction)
        pred_index_map = self.multi_hot_embed(
            masked_index, pos_items, masked_item_seq.size(-1)
        )
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)

        if self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")  # , ignore_index=0)
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [item_num H]
            logits = (torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                      + self.output_bias)  # [B mask_len item_num]
            calculated_loss = loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
            targets = (pos_items > 0).float().view(-1)
            loss_without_padding = calculated_loss * targets  # [B*mask_len]
            loss = torch.sum(loss_without_padding) / torch.sum(targets)

            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")



    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        interaction = self.reconstruct_test_data(interaction)

        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(interaction)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[test_item]  # [B]
        return scores

    def full_sort_predict(self, interaction):
        interaction = self.reconstruct_test_data(interaction)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(interaction)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[
                         : self.n_items
                         ]  # delete masked token
        scores = (
                torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        )  # [B, item_num]
        return scores
