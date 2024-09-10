# -*- coding: utf-8 -*-
# @Time   : 2022/7/19
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com

import math
import numpy
import numpy as np
import random
import torch
from copy import deepcopy
from recbole.data.interaction import Interaction, cat_interactions
from recbole.model.sequential_attribute_recommender.content_layers import create_mask_or_pad_dict


def construct_transform(config):
    """
    Transformation for batch data.
    """
    if config["transform"] is None:
        return Equal(config)
    else:
        str2transform = {
            "mask_itemseq": MaskItemSequence,
            "mask_itemseqAndAttributes": MaskItemSequenceAndAttributes,
            "inverse_itemseq": InverseItemSequence,
            "crop_itemseq": CropItemSequence,
            "reorder_itemseq": ReorderItemSequence,
            "user_defined": UserDefinedTransform,
        }
        if config["transform"] not in str2transform:
            raise NotImplementedError(
                f"There is no transform named '{config['transform']}'"
            )

        return str2transform[config["transform"]](config)


class Equal:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        return interaction


class MaskItemSequence:
    """
    Mask item sequence for training.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.MASK_ITEM_SEQ = "Mask_" + self.ITEM_SEQ
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        self.NEG_ITEMS = "Neg_" + config["ITEM_ID_FIELD"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.ft_ratio = 0 if not hasattr(config, "ft_ratio") else config["ft_ratio"]
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.MASK_INDEX = "MASK_INDEX"
        config["MASK_INDEX"] = "MASK_INDEX"
        config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
        config["POS_ITEMS"] = self.POS_ITEMS
        config["NEG_ITEMS"] = self.NEG_ITEMS
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.config = config

    def _neg_sample(self, item_set, n_items):
        item = random.randint(1, n_items - 1)
        while item in item_set:
            item = random.randint(1, n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def _append_mask_last(self, interaction, n_items, device):
        batch_size = interaction[self.ITEM_SEQ].size(0)
        pos_items, neg_items, masked_index, masked_item_sequence = [], [], [], []
        seq_instance = interaction[self.ITEM_SEQ].cpu().numpy().tolist()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].cpu().numpy().tolist()
        for instance, lens in zip(seq_instance, item_seq_len):
            mask_seq = instance.copy()
            ext = instance[lens - 1]
            mask_seq[lens - 1] = n_items
            masked_item_sequence.append(mask_seq)
            pos_items.append(self._padding_sequence([ext], self.mask_item_length))
            neg_items.append(
                self._padding_sequence(
                    [self._neg_sample(instance, n_items)], self.mask_item_length
                )
            )
            masked_index.append(
                self._padding_sequence([lens - 1], self.mask_item_length)
            )
        # [B Len]
        masked_item_sequence = torch.tensor(
            masked_item_sequence, dtype=torch.long, device=device
        ).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(
            batch_size, -1
        )
        new_dict = {
            self.MASK_ITEM_SEQ: masked_item_sequence,
            self.POS_ITEMS: pos_items,
            self.NEG_ITEMS: neg_items,
            self.MASK_INDEX: masked_index,
        }
        ft_interaction = deepcopy(interaction)
        ft_interaction.update(Interaction(new_dict))
        return ft_interaction

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        device = item_seq.device
        batch_size = item_seq.size(0)
        n_items = dataset.num(self.ITEM_ID)
        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []

        if random.random() < self.ft_ratio:
            interaction = self._append_mask_last(interaction, n_items, device)
        else:
            for instance in sequence_instances:
                # WE MUST USE 'copy()' HERE!
                masked_sequence = instance.copy()
                pos_item = []
                neg_item = []
                index_ids = []
                for index_id, item in enumerate(instance):
                    # padding is 0, the sequence is end
                    if item == 0:
                        break
                    prob = random.random()
                    if prob < self.mask_ratio:
                        pos_item.append(item)
                        neg_item.append(self._neg_sample(instance, n_items))
                        masked_sequence[index_id] = n_items
                        index_ids.append(index_id)

                masked_item_sequence.append(masked_sequence)
                pos_items.append(
                    self._padding_sequence(pos_item, self.mask_item_length)
                )
                neg_items.append(
                    self._padding_sequence(neg_item, self.mask_item_length)
                )
                masked_index.append(
                    self._padding_sequence(index_ids, self.mask_item_length)
                )

            # [B Len]
            masked_item_sequence = torch.tensor(
                masked_item_sequence, dtype=torch.long, device=device
            ).view(batch_size, -1)
            # [B mask_len]
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(
                batch_size, -1
            )
            # [B mask_len]
            masked_index = torch.tensor(
                masked_index, dtype=torch.long, device=device
            ).view(batch_size, -1)
            new_dict = {
                self.MASK_ITEM_SEQ: masked_item_sequence,
                self.POS_ITEMS: pos_items,
                self.NEG_ITEMS: neg_items,
                self.MASK_INDEX: masked_index,
            }
            interaction.update(Interaction(new_dict))
        return interaction



class MaskItemSequenceAndAttributes:
    """
    Mask item and attribute sequences for training.
    """

    def __init__(self, config):

        self.finetune = "BATCH" if not hasattr(config,"finetune") else config["finetune"]
        #NONE, BATCH, SAMPLE, ALWAYS
        self.masking_options = "ONLY_MASK" if not hasattr(config,"masking_options") else config["masking_options"]
        #RANDOM_VALUE, ONLY_MASK
        self.num_of_masked_items_choice = "actual" if not hasattr(config, "num_of_masked_items_choice") else config["num_of_masked_items_choice"]
        #max or actual or variable or variable_min

        self.minimum_masked_numbers = 1

        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.MASK_ITEM_SEQ = "mask_" + self.ITEM_SEQ
        self.POS_ITEMS = "Pos_" + config["ITEM_ID_FIELD"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.mask_ratio = config["mask_ratio"]
        self.ft_ratio = 0 if not hasattr(config, "ft_ratio") else config["ft_ratio"]
        self.mask_item_length = max(int(self.mask_ratio * self.max_seq_length),1)
        self.MASK_INDEX = "MASK_INDEX"
        config["MASK_INDEX"] = "MASK_INDEX"
        config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
        config["POS_ITEMS"] = self.POS_ITEMS
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.item_config = config["items"]
        self.config = config
        self.masking_tokens_dict = None

    def create_mask_dict(self, item_attributes, dataset):
        mask_dict = {}
        if item_attributes is not None:
            for attribute_name, attribute_infos in item_attributes["attributes"].items():
                if attribute_infos.get("mask", None) is not None:
                    mask_dict[attribute_name] = attribute_infos["mask"]
                else:
                    mask_dict[attribute_name] = len(dataset.field2token_id[attribute_name])
                mask_len = attribute_infos.get("len", None)
                if mask_len is not None and mask_len != 1:
                    mask_dict[attribute_name] = [mask_dict[attribute_name]] * mask_len
            if item_attributes.get("item_id_type_settings", None) is not None:
                mask_dict[item_attributes["item_id_type_settings"]["name"]] = 0
        return mask_dict

    def __call__(self, dataset, interaction):

        batched_item_sequences = interaction[self.ITEM_SEQ]
        batched_masked_item_sequences = batched_item_sequences.clone()
        batched_sequence_lengths = interaction[self.ITEM_SEQ_LEN]
        if torch.all(batched_sequence_lengths < 2):
            raise ValueError("batched_sequence_lengths contains len < 2, line 250")
        device = batched_item_sequences.device
        batch_size = batched_item_sequences.size(0)
        item_masking_token = dataset.num(self.ITEM_ID)

        # Create masks tokens for attributes once
        if self.masking_tokens_dict is None:
            self.masking_tokens_dict = self.create_mask_dict(self.item_config, dataset)

        batched_masked_attribute_sequences = {} #Dict with all attributes to be masked

        if self.item_config is not None:
            if self.item_config.get("attributes") is not None:
                for (attribute_name, infos) in self.item_config["attributes"].items():
                    batched_masked_attribute_sequences[attribute_name] = interaction[attribute_name + "_list"].clone()
            if self.item_config.get("item_id_type_settings") is not None:
                item_id_name = self.item_config["item_id_type_settings"]["name"]
                batched_masked_attribute_sequences[item_id_name] = interaction[item_id_name + "_list"].clone()

        final_mask_len = self.mask_item_length
        if self.finetune == "BATCH":
            finetune_random_number = random.random()
        if self.finetune == "ALWAYS":
            final_mask_len = self.mask_item_length + 1

        batch_mask_index_ids = torch.zeros([batch_size, final_mask_len], dtype=torch.long, device=device)
        batch_pos_items = torch.zeros([batch_size, final_mask_len], dtype=torch.long, device=device)

        for sample_id in range(batch_size):
            original_item_sequence = batched_masked_item_sequences[sample_id]
            masked_item_sequence = batched_masked_item_sequences[sample_id]

            if self.finetune == "SAMPLE":
                finetune_random_number = random.random()

            if self.finetune in ["BATCH","SAMPLE"] and finetune_random_number < self.ft_ratio:
                number_of_masked_items = 1
                mask_index_ids = torch.tensor([batched_sequence_lengths[sample_id] - 1], dtype=torch.long, device=device)
                positive_items = masked_item_sequence[mask_index_ids]
                masked_item_sequence[mask_index_ids] = item_masking_token
                if 0 in positive_items:
                    raise ValueError("0 in tensor positive_items, line 288")
            else:
                #Determine the number of masked items
                if self.num_of_masked_items_choice == "actual":
                    number_of_masked_items = max(int(self.mask_ratio * batched_sequence_lengths[sample_id]), 1)
                    base_sequence_len = batched_sequence_lengths[sample_id]
                elif self.num_of_masked_items_choice == "max":
                    number_of_masked_items = self.mask_item_length
                    base_sequence_len = self.max_seq_length
                elif self.num_of_masked_items_choice == "variable":
                    number_of_masked_items = random.randint(self.minimum_masked_numbers, self.mask_item_length)
                    base_sequence_len = self.max_seq_length
                elif self.num_of_masked_items_choice == "variable_min":
                    max_number = max(int(self.mask_ratio * batched_sequence_lengths[sample_id]), 1)
                    number_of_masked_items = random.randint(self.minimum_masked_numbers, max_number)
                    base_sequence_len = batched_sequence_lengths[sample_id]
                else:
                    raise ValueError("num_of_masked_items_choice must be one of 'actual', 'max' or 'variable_min' 'variable'")

                #Determine IDs of masked Items
                mask_index_ids = torch.randperm(base_sequence_len, dtype=torch.long, device=device)[:number_of_masked_items]
                positive_items = masked_item_sequence[mask_index_ids]
                if 0 in positive_items:
                    torch.set_printoptions(profile="full")
                    print("original_item_sequence: ")
                    print(original_item_sequence)
                    print("mask_index_ids: ")
                    print(mask_index_ids)
                    print("positive_items: ")
                    print(positive_items)
                    torch.set_printoptions(profile="default")  # reset
                    raise ValueError("0 in tensor positive_items, line 309")

                if self.masking_options == "ONLY_MASK":
                    masked_item_sequence[mask_index_ids] = item_masking_token

                elif self.masking_options == "RANDOM_VALUE":
                    for current_mask_index in mask_index_ids:
                        prop = random.random()
                        if prop < 0.8:
                            masked_item_sequence[current_mask_index] = item_masking_token
                        elif prop < 0.9:
                            masked_item_sequence[current_mask_index] = random.randint(1, item_masking_token - 1)
                        elif prop <1.0:
                           ... # keep Value

                if self.finetune == "ALWAYS":
                    if batched_sequence_lengths[sample_id] -1 not in mask_index_ids:
                        number_of_masked_items += 1
                        mask_index_ids = torch.concat([mask_index_ids,
                                                       torch.tensor([batched_sequence_lengths[sample_id] - 1], dtype=torch.long, device=device)])
                        positive_items = torch.concat([positive_items,torch.tensor([masked_item_sequence[batched_sequence_lengths[sample_id] - 1]], dtype=torch.long, device=device)])
                        masked_item_sequence[mask_index_ids] = item_masking_token
                        if 0 in positive_items:
                            raise ValueError("0 in tensor positive_items, line 332")

            #Pad to the max mask length with zeros
            mask_index_ids = torch.concat([torch.zeros(final_mask_len - number_of_masked_items, dtype=torch.long,
                                                       device=device), mask_index_ids])
            batch_mask_index_ids[sample_id] = mask_index_ids
            batch_pos_items[sample_id] = torch.concat([torch.zeros(final_mask_len - number_of_masked_items,
                                                                   dtype=torch.long, device=device), positive_items])

            if torch.sum(batch_pos_items) == 0:
                torch.set_printoptions(profile="full")
                print("batched_item_sequences: ")
                print(batched_item_sequences)
                print("batch_pos_items: ")
                print(batch_pos_items)
                torch.set_printoptions(profile="default")  # reset
                raise ValueError("sum(batch_pos_items) = 0, line 339")

            for (attribute_name, attribute_sequences) in batched_masked_attribute_sequences.items():
                attribute_sequence_sample = attribute_sequences[sample_id]
                mask_token = self.masking_tokens_dict[attribute_name]
                attribute_sequence_sample[mask_index_ids] = torch.tensor(mask_token, dtype=attribute_sequences.dtype,
                                                                         device=device)

        masked_entries_dict = {
            self.MASK_ITEM_SEQ: batched_masked_item_sequences,
            self.MASK_INDEX: batch_mask_index_ids,
            self.POS_ITEMS: batch_pos_items,
        }
        for (k, attribute_sequences) in batched_masked_attribute_sequences.items():
            masked_entries_dict[("mask_" + k + "_list")] = attribute_sequences
        interaction.update(Interaction(masked_entries_dict))
        return interaction


class InverseItemSequence:
    """
    inverse the seq_item, like this
        [1,2,3,0,0,0,0] -- after inverse -->> [0,0,0,0,1,2,3]
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.INVERSE_ITEM_SEQ = "Inverse_" + self.ITEM_SEQ
        config["INVERSE_ITEM_SEQ"] = self.INVERSE_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        item_seq = item_seq.cpu().numpy()
        item_seq_len = item_seq_len.cpu().numpy()
        new_item_seq = []
        for items, length in zip(item_seq, item_seq_len):
            item = list(items[:length])
            zeros = list(items[length:])
            seqs = zeros + item
            new_item_seq.append(seqs)
        inverse_item_seq = torch.tensor(new_item_seq, dtype=torch.long, device=device)
        new_dict = {self.INVERSE_ITEM_SEQ: inverse_item_seq}
        interaction.update(Interaction(new_dict))
        return interaction


class CropItemSequence:
    """
    Random crop for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.CROP_ITEM_SEQ_LEN = self.CROP_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.crop_eta = config["eta"]
        config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
        config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        crop_item_seq_list, crop_item_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            crop_len = math.floor(length * self.crop_eta)
            crop_begin = random.randint(0, length - crop_len)
            crop_item_seq = np.zeros(seq.shape[0])
            if crop_begin + crop_len < seq.shape[0]:
                crop_item_seq[:crop_len] = seq[crop_begin : crop_begin + crop_len]
            else:
                crop_item_seq[:crop_len] = seq[crop_begin:]
            crop_item_seq_list.append(
                torch.tensor(crop_item_seq, dtype=torch.long, device=device)
            )
            crop_item_seqlen_list.append(
                torch.tensor(crop_len, dtype=torch.long, device=device)
            )
        new_dict = {
            self.CROP_ITEM_SEQ: torch.stack(crop_item_seq_list),
            self.CROP_ITEM_SEQ_LEN: torch.stack(crop_item_seqlen_list),
        }
        interaction.update(Interaction(new_dict))
        return interaction


class ReorderItemSequence:
    """
    Reorder operation for item sequence.
    """

    def __init__(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.reorder_beta = config["beta"]
        config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ

    def __call__(self, dataset, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        reorder_seq_list = []

        for seq, length in zip(item_seq, item_seq_len):
            reorder_len = math.floor(length * self.reorder_beta)
            reorder_begin = random.randint(0, length - reorder_len)
            reorder_item_seq = seq.cpu().detach().numpy().copy()

            shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
            random.shuffle(shuffle_index)
            reorder_item_seq[reorder_begin : reorder_begin + reorder_len] = (
                reorder_item_seq[shuffle_index]
            )

            reorder_seq_list.append(
                torch.tensor(reorder_item_seq, dtype=torch.long, device=device)
            )
        new_dict = {self.REORDER_ITEM_SEQ: torch.stack(reorder_seq_list)}
        interaction.update(Interaction(new_dict))
        return interaction


class UserDefinedTransform:
    def __init__(self, config):
        pass

    def __call__(self, dataset, interaction):
        pass
