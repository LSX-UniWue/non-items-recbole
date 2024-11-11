# -*- coding: utf-8 -*-
# @Time    : 2022/02/22 19:32
# @Author  : Peilin Zhou, Yueqi Xie
# @Email   : zhoupl@pku.edu.cn
r"""
SASRecD
################################################

Reference:
    Yueqi Xie and Peilin Zhou et al. "Decouple Side Information Fusion for Sequential Recommendation"
    Submited to SIGIR 2022.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
import copy

from recbole.model.dif_layers import DIFTransformerEncoder
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, embed_attributes, \
    create_mask_or_pad_dict, concat_user_embeddings, merge_user_embeddings


class DIFSRAttr(SequentialRecommender):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation. This implementation is limited to the fusion operation and
    doesn't include additional training objectives
    """

    def __init__(self, config, dataset):
        super(DIFSRAttr, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']
        self.num_feature_field = len(config['items']['attributes'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.fusion_type = config['items']["attribute_fusion"]

        #self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)


        self.item_attributes = config["items"]
        self.user_attributes = config["users"]
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.attribute_hidden_size)
        self.user_attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.user_attributes,
                                                                     self.attribute_hidden_size)
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")
        self.user_fusion = None if self.user_attributes is None else self.user_attributes.get("user_fusion", "concat")
        if self.user_fusion == "concat":
            self.final_seq_length = self.max_seq_length + 1
        else:
            self.final_seq_length = self.max_seq_length
        self.position_embedding = nn.Embedding(self.final_seq_length, self.hidden_size)
        self.trm_encoder = DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            feat_num=len(self.item_attributes["attributes"]),
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.final_seq_length
        )

        #self.n_attributes = {}
        #for attribute in self.selected_features:
        #    self.n_attributes[attribute] = len(dataset.field2token_id[attribute])
        #if self.attribute_predictor == 'MLP':
        #    self.ap = nn.Sequential(nn.Linear(in_features=self.hidden_size,
        #                                               out_features=self.hidden_size),
        #                                     nn.BatchNorm1d(num_features=self.hidden_size),
        #                                     nn.ReLU(),
        #                                     # final logits
         #                                    nn.Linear(in_features=self.hidden_size,
        #                                               out_features=self.n_attributes)
        #                                     )
        #elif self.attribute_predictor == 'linear':
         #   self.ap = nn.ModuleList(
        #        [copy.deepcopy(nn.Linear(in_features=self.hidden_size, out_features=self.n_attributes[_]))
         #        for _ in self.selected_features])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            #self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb = self.item_embedding(item_seq)

        embedded_features = embed_attributes(interaction, self.item_attributes, self.attribute_embeddings,
                                             pad_values=self.pad_dict)
        embedded_user_features = embed_attributes(interaction, self.user_attributes, self.user_attribute_embeddings)
        if self.user_fusion == "concat":
            item_emb = concat_user_embeddings(self.user_attributes, embedded_user_features, item_emb)
            user_mask = torch.zeros((item_seq.size(0), 1), device=item_seq.device, dtype=item_seq.dtype)
            item_seq = torch.concat((user_mask, item_seq), dim=1)
            user_feature_mask = torch.zeros((item_seq.size(0), 1, self.attribute_hidden_size), device=item_seq.device, dtype= item_seq.dtype)
            for feature in embedded_features:
                embedded_features[feature] = torch.cat((user_feature_mask,embedded_features[feature]),dim=1)

        feature_emb = list(embedded_features.values())

        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        trm_output = trm_output[-1]

        if self.user_fusion == "post_merge":
            trm_output = merge_user_embeddings(self.user_attributes, embedded_user_features, sequence=trm_output)
        if self.user_fusion == "concat":
            trm_output = trm_output[:, 1:, :]
        seq_output = self.gather_indexes(trm_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        seq_output = self.forward(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
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
            #if self.attribute_predictor!='' and self.attribute_predictor!='not':
            #    loss_dic = {'item_loss':loss}
            #    attribute_loss_sum = 0
            #    for i, a_predictor in enumerate(self.ap):
            #        attribute_logits = a_predictor(seq_output)
            #        attribute_labels = interaction.interaction[self.selected_features[i]]
            #        attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.n_attributes[
            #            self.selected_features[i]])

            #        if len(attribute_labels.shape) > 2:
            #            attribute_labels = attribute_labels.sum(dim=1)
            #        attribute_labels = attribute_labels.float()
            #        attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels)
            #        attribute_loss = torch.mean(attribute_loss[:, 1:])
            #        loss_dic[self.selected_features[i]] = attribute_loss
            #    if self.num_feature_field == 1:
            #        total_loss = loss + self.lamdas[0] * attribute_loss
                    # print('total_loss:{}\titem_loss:{}\tattribute_{}_loss:{}'.format(total_loss, loss,self.selected_features[0],attribute_loss))
           #     else:
            #        for i,attribute in enumerate(self.selected_features):
            #            attribute_loss_sum += self.lamdas[i] * loss_dic[attribute]
            #        total_loss = loss + attribute_loss_sum
             #       loss_dic['total_loss'] = total_loss
                    # s = ''
                    # for key,value in loss_dic.items():
                    #     s += '{}_{:.4f}\t'.format(key,value.item())
                    # print(s)
            #else:
            total_loss = loss
            return total_loss

    def predict(self, interaction):
        seq_output = self.forward(interaction)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        seq_output = self.forward(interaction)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores