import copy
import math
import torch
import recbole.model.layers as layers
from recbole.model.sequential_attribute_recommender.content_layers import create_attribute_embeddings
from recbole.model.sequential_recommender import SASRec
from torch import nn

class SASRecNOVA(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follow the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecNOVA, self).__init__(config, dataset)
        self.activation_func = config["fusion_activation_function"]
        self.fusion_type = config["items"]["attribute_fusion"]
        self.residual = bool(config["residual"])
        self.ids_to_attr_input = bool(config["ids_to_attr_input"])
        self.norm_attr_input = bool(config["norm_attr_input"])
        self.dropout_attr_input = bool(config["dropout_attr_input"])

        self.item_attributes = config["items"]
        self.embedding_size = config["hidden_size"]
        self.attribute_embeddings = create_attribute_embeddings(dataset.field2token_id, self.item_attributes,
                                                                self.embedding_size, None)

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

        # number of features
        # +1 because positional embedding + (optional) +1 item_seq embedding
        feature_count = len(self.item_attributes["attributes"]) + 1
        if self.ids_to_attr_input:
            feature_count = feature_count + 1

        self.linear_Layer_concat = nn.Linear(self.hidden_size * feature_count, self.hidden_size)

        # parameters for gated fusion
        self.original_gate_param = nn.Parameter(torch.ones(self.hidden_size, 1), True)
        self.per_feature_per_entry_param = nn.Parameter(torch.ones(self.hidden_size, self.hidden_size), True)

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

        self.apply(self._init_weights)

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
        if fusion_type == "concat":
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
            attention_mask = self.get_attention_mask(item_seq, bidirectional=False)
            attended_features_list = []
            for i in range(len(features_emb_list)):
                ith_hidden_states = self.attention_block_list[i](item_emb, features_emb_list[i], attention_mask)
                ith_attended_feature = torch.mul(features_emb_list[i], ith_hidden_states)
                attended_features_list.append(ith_attended_feature)
            attended_features_matrix = torch.stack(attended_features_list, 0)
            fused_out = torch.sum(attended_features_matrix, 0)
            return fused_out


    def get_embedded_feature_list(self, interaction):
        features_emb_list = []
        for (feature_name, infos) in self.item_attributes['attributes'].items():
            feature_tensor = interaction[feature_name + "_list"]
            if infos["embedding_type"] == "float":
                feature_tensor = torch.unsqueeze(feature_tensor, 2)
            feature_emb = self.attribute_embeddings[feature_name](feature_tensor)
            features_emb_list.append(feature_emb)
        return features_emb_list


    def forward(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        features_emb_list = self.get_embedded_feature_list(interaction)

        features_emb_list.append(position_embedding)
        if self.ids_to_attr_input:
            features_emb_list.append(item_emb)

        fused_emb_features = self.fusion(item_seq, item_emb, features_emb_list, self.fusion_type)

        if self.norm_attr_input:
            fused_emb_features = self.LayerNorm(fused_emb_features)
        if self.dropout_attr_input:
            fused_emb_features = self.dropout(fused_emb_features)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            fused_emb_features, item_emb, extended_attention_mask, output_all_encoded_layers=True
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



"""

New layer definitions for NOVASasRec

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


