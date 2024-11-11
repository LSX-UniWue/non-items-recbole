import copy
import math
import recbole.model.layers as layers
import torch
from torch import nn

from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings, create_mask_or_pad_dict
from recbole.model.abstract_recommender import SequentialRecommender



class BERT4RecNOVA(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BERT4RecNOVA, self).__init__(config, dataset)

        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.item_attributes = config["items"]
        self.token_ids = dataset.field2token_id
        self.pooling_mode = config["pooling_mode"]
        self.selected_features = config["selected_features"]

        # Modifications
        self.fusion_type = config["items"]["attribute_fusion"]
        self.activation_func = config["fusion_activation_function"]
        self.residual = bool(config["residual"])
        self.ids_to_attr_input = bool(config["ids_to_attr_input"])
        self.norm_attr_input = bool(config["norm_attr_input"])
        self.dropout_attr_input = bool(config["dropout_attr_input"])

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        #self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1

        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.hidden_size, True)

        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        self.trm_encoder = NovaTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.dataset = dataset
        self.config = config

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # number of features
        # +1 because positional embedding + (optional) +1 item_seq embedding
        feature_count = len(self.item_attributes["attributes"]) + 1
        if self.ids_to_attr_input:
            feature_count = feature_count + 1

        self.linear_Layer_concat = nn.Linear(self.hidden_size * feature_count, self.hidden_size)

        # parameters for gated fusion
        self.original_gate_param = nn.Parameter(torch.ones(self.hidden_size, 1), True)
        self.per_feature_per_entry_param = nn.Parameter(torch.ones(self.hidden_size, self.hidden_size), True)

        self.mask_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="mask")
        self.pad_dict = create_mask_or_pad_dict(self.item_attributes, dataset, logger=self.logger, mask_or_pad="pad")

        # attention for fusion
        if self.fusion_type == "attention_fusion":

            attention_block = FusionMultiHeadAttention(
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                layer_norm_eps=self.layer_norm_eps
            )
            self.attention_block_list = nn.ModuleList([copy.deepcopy(attention_block) for _ in range(feature_count)])


        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

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
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.initializer_range)

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
            if self.item_attributes["item_id_type_settings"] is not None:
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

    def fusion(self, item_seq, item_emb, features_emb_list, fusion_type: str):
        if self.activation_func == "sigmoid":
            activation = nn.Sigmoid()
        elif self.activation_func == "relu":
            activation = nn.ReLU()
        else:
            NotImplementedError("fusion_activation_function must be set in config to [sigmoid, relu]!")
        if fusion_type == "sum":
                tensors = torch.stack(features_emb_list, 2)
                fused_tensor = torch.sum(tensors, 2)
                return fused_tensor
        if fusion_type =="concat":
                fused_tensor = torch.cat(features_emb_list, 2)
                out = self.linear_Layer_concat(fused_tensor)
                return out
        if fusion_type == "gating_original":
                features_matrix = torch.stack(features_emb_list, 2)
                gate = activation(torch.matmul(features_matrix, self.original_gate_param))
                gated_features = []
                for i in range(len(features_emb_list)):
                    gate_entry = gate[:, :, i]
                    gated_feature = torch.mul(gate_entry, features_emb_list[i])
                    gated_features.append(gated_feature)
                gated_features_tensor = torch.stack(gated_features, 2)
                fused_out = torch.sum(gated_features_tensor, 2)
                return fused_out
        if fusion_type == "gating_per_feature_per_entry":
                features_matrix = torch.stack(features_emb_list, 2)
                gate = activation(torch.matmul(features_matrix, self.per_feature_per_entry_param))
                gated_features = []
                for i in range(len(features_emb_list)):
                    gate_entry = gate[:, :, i, :]
                    gated_feature = torch.mul(gate_entry, features_emb_list[i])
                    gated_features.append(gated_feature)
                gated_features_tensor = torch.stack(gated_features, 2)
                fused_out = torch.sum(gated_features_tensor, 2)
                return fused_out
        if fusion_type == "attention_fusion":
                attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
                attended_features_list = []
                for i in range(len(features_emb_list)):
                    ith_hidden_states = self.attention_block_list[i](item_emb, features_emb_list[i], attention_mask)
                    ith_attended_feature = torch.mul(features_emb_list[i], ith_hidden_states)
                    attended_features_list.append(ith_attended_feature)
                attended_features_matrix = torch.stack(attended_features_list, 0)
                fused_out = torch.sum(attended_features_matrix, 0)
                return fused_out
    def get_embedded_feature_list(self, interaction, masked = False):
        features_emb_list = []
        for (feature_name, infos) in self.item_attributes['attributes'].items():
            if masked:
                feature_tensor = interaction["mask_" + feature_name + "_list"]
            else:
                feature_tensor = interaction[feature_name + "_list"]
            if infos["embedding_type"] == "float":
                feature_tensor = torch.unsqueeze(feature_tensor, 2)
            feature_emb = self.attribute_embeddings[feature_name](feature_tensor)
            features_emb_list.append(feature_emb)
        return features_emb_list


    def forward(self, interaction):
        item_seq = interaction[self.MASK_ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        features_emb_list = self.get_embedded_feature_list(interaction, masked=True)

        features_emb_list.append(position_embedding)
        if self.ids_to_attr_input:
            features_emb_list.append(item_emb)

        fused_emb_features = self.fusion(item_seq, item_emb, features_emb_list, self.fusion_type)

        if self.norm_attr_input:
            fused_emb_features = self.LayerNorm(fused_emb_features)
        if self.dropout_attr_input:
            fused_emb_features = self.dropout(fused_emb_features)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            fused_emb_features, item_emb, extended_attention_mask, output_all_encoded_layers=True
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
            if torch.sum(targets) == 0:
                raise ValueError("sum(targets) = 0")
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

"""
New layer definitions for NOVABert

"""
class NovaMultiHeadAttention(nn.Module):

    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(NovaMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor_all, input_tensor_ids, attention_mask):

        """new Q,K,V computation"""
        mixed_query_layer = self.query(input_tensor_all)
        mixed_key_layer = self.key(input_tensor_all)
        mixed_value_layer = self.value(input_tensor_ids)

        """rest same as in base MultiHeadAttention"""
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor_ids)

        return hidden_states


class NovaTransformerLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
    ):
        super(NovaTransformerLayer, self).__init__()
        self.multi_head_attention = NovaMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = layers.FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, residual_tensor_sideinfo, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(residual_tensor_sideinfo, hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class NovaTransformerEncoder(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(NovaTransformerEncoder, self).__init__()
        layer = NovaTransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, residual_tensor_sideinfo, hidden_states, attention_mask, output_all_encoded_layers=True):

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(residual_tensor_sideinfo, hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# New attention for gating

class FusionMultiHeadAttention(nn.Module):

    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(FusionMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, id_tensor, feature_tensor, attention_mask):

        mixed_query_layer = self.query(id_tensor)
        mixed_key_layer = self.key(feature_tensor)
        mixed_value_layer = self.value(feature_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        # TO DO ??(hidden_states + id_tensor + feature_tensor)??
        hidden_states = self.LayerNorm(hidden_states + id_tensor)

        return hidden_states
