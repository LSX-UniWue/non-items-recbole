import numpy as np
import torch
from torch import nn

from recbole.model.layers import FMEmbedding, FLEmbedding
from recbole.utils import FeatureType


def _build_embedding_type(embedding_type: str,
                          vocab_size: int,
                          hidden_size: int
                          ) -> nn.Module:
    return {
        'std_embedding': nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size),
        'multi_hot': MultiHotNScale(vocab_size=vocab_size,
                                         embed_size=hidden_size)
    }[embedding_type]


def create_mask_or_pad_dict(item_attributes, dataset, logger, mask_or_pad):
    mask_dict = {}
    if item_attributes is not None:
        if item_attributes.get("attributes", None) is not None:
            for attribute_name, attribute_infos in item_attributes["attributes"].items():
                if attribute_infos.get(mask_or_pad, None) is not None:
                    mask_dict[attribute_name] = attribute_infos[mask_or_pad]
                else:
                    try:
                        if mask_or_pad == "mask":
                            mask_dict[attribute_name] = len(dataset.field2token_id[attribute_name])
                            logger.info(
                                "No mask token provided for attribute {}. Using the id {} as mask token.".format(
                                    attribute_name, mask_dict[attribute_name]))
                        if mask_or_pad == "pad":
                            mask_dict[attribute_name] = 0
                            logger.info(
                                "No pad token provided for attribute {}. Using the id {} as pad token.".format(
                                    attribute_name, mask_dict[attribute_name]))
                    except KeyError:
                        logger.error("No {} token provided for attribute {}.".format(mask_or_pad,attribute_name))
                mask_len = attribute_infos.get("len", None)
                if mask_len is not None and mask_len != 1:
                    mask_dict[attribute_name] = [mask_dict[attribute_name]] * mask_len
        if item_attributes.get("item_id_type_settings", None) is not None and mask_or_pad == "mask":
             mask_dict[item_attributes["item_id_type_settings"]["name"]] = 0
    return mask_dict


def create_attribute_embeddings(field2token_id, attributes, hidden_size, masking=False):
    attribute_embeddings = {}
    if attributes is not None:
        if attributes.get("attributes") is not None:
            for attribute_name, attribute_infos in attributes["attributes"].items():
                embedding_type = attribute_infos["embedding_type"]
                if embedding_type in ["std_embedding", "multihot"]:
                    input_size = len(field2token_id[attribute_name])
                    if masking:
                        input_size = input_size + 1
                    if embedding_type == "std_embedding":
                        attribute_embeddings[attribute_name] = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
                    if embedding_type == "multihot":
                        attribute_embeddings[attribute_name] = MultiHotNScale(vocab_size=input_size, embed_size=hidden_size)
                elif embedding_type == "float":
                    attribute_embeddings[attribute_name] = VectorNormNScale(1, hidden_size)
                elif embedding_type == "vector":
                    size = attribute_infos["len"]
                    attribute_embeddings[attribute_name] = VectorNormNScale(size,hidden_size)
    return nn.ModuleDict(attribute_embeddings)

def embed_attributes(interaction, attributes_config, attribute_embeddings, use_masked_sequence=False, pad_values={}):
    # embed the attribute embeddings
    embedded_features = {}
    if attributes_config is not None and attributes_config.get("attributes") is not None:
        skip_item = attributes_config.get("attributes_listpage_only", False)
        for feature in attributes_config.get("attributes"):
            if use_masked_sequence:
                additional_metadata = interaction["mask_"+feature + "_list"]
            else:
                additional_metadata = interaction[feature + "_list"]
            if skip_item: #only use attributes to represent non items
                if use_masked_sequence:
                    item_id_type = interaction["mask_" + attributes_config["item_id_type_settings"]["name"] + "_list"]
                else:
                    item_id_type = interaction[attributes_config["item_id_type_settings"]["name"] + "_list"]
                additional_metadata[item_id_type == 1] = torch.tensor(pad_values[feature], device=additional_metadata.device).type(additional_metadata.dtype)
            if attributes_config["attributes"][feature]["embedding_type"] == "float":
                additional_metadata = torch.unsqueeze(additional_metadata, 2)
            embedded_features[feature] = attribute_embeddings[feature](additional_metadata)
    return embedded_features


def merge_embedded_item_features(embedded_features, attributes_config, item_seq_emb):
    if attributes_config is not None and attributes_config.get("attributes") is not None:
        merge = attributes_config.get("attribute_fusion", None)
        if merge == "sum":
            for feature in attributes_config["attributes"]:
                item_seq_emb = item_seq_emb + embedded_features[feature]
        if merge == "multiply":
            for feature in attributes_config["attributes"]:
                item_seq_emb = item_seq_emb * embedded_features[feature]
    return item_seq_emb


def concat_user_embeddings(user_attributes, embedded_user_features, item_seq_emb):
    if user_attributes is not None:
        merge = user_attributes.get("attribute_fusion", None)
        user_embedding_list = list(embedded_user_features.values())
        user_embedding = user_embedding_list[0][:, 0:1, :]  # get the first user embedding
        for i in range(1, len(user_embedding_list) - 1):
            if merge == "sum":
                user_embedding = user_embedding + user_embedding_list[i][:, 0:1, :]
            if merge == "multiply":
                user_embedding = user_embedding * user_embedding_list[i][:, 0:1, :]
        item_seq_emb = torch.concat((user_embedding, item_seq_emb), dim=1)
    return item_seq_emb



class ContentVectorMaskAndScale(nn.Module):
    def __init__(self, input_size: int, embed_size: int, item_mask_token: int):
        super().__init__()
        self.item_mask_token = item_mask_token
        self.linear = nn.Linear(input_size, embed_size)
        self.trained_mask = nn.Parameter(torch.Tensor(input_size))
        self.embedding_norm = nn.LayerNorm(input_size)
        nn.init.normal_(self.trained_mask, mean=1, std=0.5)

    def forward(self,
                content_sequence: torch.Tensor,
                item_sequence: torch.Tensor
                ) -> torch.Tensor:
        mask_indices = (item_sequence == self.item_mask_token).unsqueeze(-1)
        sequence = torch.where(mask_indices, self.trained_mask, content_sequence)
        sequence = self.embedding_norm(sequence)
        sequence = self.linear(sequence)

        return sequence

class VectorNormNScale(nn.Module):

    def __init__(self, input_size: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, embed_size)
        self.embedding_norm = nn.LayerNorm(input_size)
    def forward(self,
                content_sequence: torch.Tensor) -> torch.Tensor:
        content_sequence = self.embedding_norm(content_sequence)
        content_sequence = self.linear(content_sequence)
        return content_sequence

class MultiHotNScale(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        """
        Creates MultiHot Encoding followed by a linear layer to scale the embedding
        :param vocab_size: Vocabulary size
        :param embed_size: Embedding size
        """
        super().__init__()
        self.linear = nn.Linear(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self,
                content_input: torch.Tensor
                ) -> torch.Tensor:
        """
        :param content_input: a tensor containing ids to be multi-hot encoded
        :return:
        """
        # the input is a sequence of content ids without any order
        # so we convert them into a multi-hot encoding
        multi_hot = torch.nn.functional.one_hot(content_input, self.vocab_size).sum(2).float()
        # 0 is the padding category, so zero it out
        multi_hot[:, :, 0] = 0
        return self.linear(multi_hot)


