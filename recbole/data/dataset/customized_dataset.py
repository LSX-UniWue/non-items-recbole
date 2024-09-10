# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import numpy as np
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset, Dataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][:length] = (
                            self.neg_item_list[index]
                        )

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class AttrSequentialDataset(SequentialDataset):
    """
    AttrSequentialDataset is based on SequentialDataset. In difference to SequentialDataset, it will not only join
    single user/item features for the target into one interaction, but also join item attributes for the whole sequence.
    This might effect efficiency, but loading attributes is neccessary here.
    """

    def __init__(self, config):
        super().__init__(config)
        self.iid_list_field = self.config["ITEM_ID_FIELD"] + "_list"

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.

        """

        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        # For sequence features, join item attributes
        if self.item_feat is not None and self.iid_list_field in df:
            iid_list = [df[self.iid_list_field]]
            iid_feature_list = self.item_feat[iid_list]
            for k in iid_feature_list.interaction:
                df.interaction[k + "_list"] = iid_feature_list.interaction[k]
        return df


class BenchAttrSequentialDataset(SequentialDataset):
    """
    BenchAttrSequentialDataset is based on SequentialDataset. In difference to SequentialDataset, it will not only join
    single user/item features for the target into one interaction, but also join item attributes for the whole sequence.
    It also uses benchmark_files, but in the formating of normal inter files to load a fix train/test/val split.
    """

    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super(SequentialDataset, self).__init__(config)
        self.iid_list_field = self.config["ITEM_ID_FIELD"] + "_list"

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        datasets = None
        self._change_feat_format()
        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]

        # Augmentation only for Training
        self.masked_training = False if not hasattr(self.config, "masked_training") else self.config["masked_training"]
        self.train_subsequences = False if not hasattr(self.config, "train_subsequences") else self.config[
            "train_subsequences"]
        self.test_subsequences = False if not hasattr(self.config, "test_subsequences") else self.config[
            "test_subsequences"]
        self.subsequences_end_with_items = False if not hasattr(self.config, "subsequences_end_with_items") else \
        self.config["subsequences_end_with_items"]

        datasets[0].data_augmentation(subsequences=self.train_subsequences, masked_training=self.masked_training,
                                      subsequences_end_with_items=self.subsequences_end_with_items)
        datasets[1].data_augmentation(subsequences=self.test_subsequences,
                                      subsequences_end_with_items=self.subsequences_end_with_items)
        datasets[2].data_augmentation(subsequences=self.test_subsequences,
                                      subsequences_end_with_items=self.subsequences_end_with_items)

        self.inter_feat = datasets[0].inter_feat  # So there's something the calculate the flops for logging
        return datasets

    def data_augmentation(self, subsequences=False, masked_training=False, subsequences_end_with_items=False):
        """Augmentation processing for sequential dataset, with the option to generate subsequences.

        Sequence ``<i1, i2, i3, i4>`` will be split into subsequences of the form:
        ``<i1> | i2``, ``<i1, i2> | i3``, ``<i1, i2, i3> | i4``

        For masked training, the  ``<i1, i2, i3, i4>`` will be split as follows, with a dummy target that is not used:
        ``<i1, i2> | i2 ``, ``<i1, i2, i3> | i3``, ``<i1, i2, i3, i4> | i4``
        This allows all models to see the same target items while training.

        """
        if masked_training:
            self.logger.info("Masked Training: Dummy Targets generated for training data.")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        seq_end = 0
        t_index = 0

        if subsequences:
            if subsequences_end_with_items is True:
                self.logger.info("Subsequences will end with items.")
                item_id_type = self.config["items"]["item_id_type_settings"]["name"]
                item_id_type_feature = self.inter_feat[item_id_type].numpy()
            elif subsequences_end_with_items is False:
                self.logger.info("Subsequences generated.")
                # Treat everything as a target
                item_id_type_feature = np.ones(self.inter_feat[self.iid_field].numpy().shape)

            for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
                if last_uid is not None and last_uid != uid:
                    # We have collected a whole session from seq_start to t_index
                    self.create_subsequences(item_id_type_feature, item_list_index, item_list_length, last_uid,
                                             masked_training, max_item_list_len, seq_start, t_index, target_index,
                                             uid_list)
                    seq_start = i
                last_uid = uid
                t_index = i
            self.create_subsequences(item_id_type_feature, item_list_index, item_list_length, last_uid,
                                     masked_training, max_item_list_len, seq_start, t_index, target_index, uid_list)
        else:
            for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
                if last_uid is not None and last_uid != uid:
                    uid_list.append(last_uid)
                    if masked_training:
                        item_list_index.append(slice(seq_start + 1, t_index + 1))
                    else:
                        item_list_index.append(slice(seq_start, t_index))
                    target_index.append(t_index)
                    item_list_length.append(t_index - seq_start)
                    seq_start = i
                else:
                    if t_index - seq_start >= max_item_list_len:
                        seq_start += 1
                last_uid = uid
                t_index = i
                seq_end = t_index - 1

            uid_list.append(last_uid)
            if masked_training:
                item_list_index.append(slice(seq_start + 1, t_index + 1))
            else:
                item_list_index.append(slice(seq_start, t_index))
            target_index.append(t_index)
            item_list_length.append(t_index - seq_start)

        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)

        new_data = self.inter_feat[target_index]
        new_dict = {self.item_list_length_field: torch.tensor(item_list_length), }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = ((new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len)
                if (self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ] and field in self.config[
                    "numerical_features"]):
                    shape += (2,)
                new_dict[list_field] = torch.zeros(shape, dtype=self.inter_feat[field].dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]
        new_data.update(Interaction(new_dict))

        self.inter_feat = new_data

    def create_subsequences(self, item_id_type_feature, item_list_index, item_list_length, last_uid, masked_training,
                            max_item_list_len, seq_start, t_index, target_index, uid_list):
        item_indices_session = np.where(item_id_type_feature[seq_start:t_index+1] == 1)[0]
        if len(item_indices_session) == 0:
            self.logger.warning(f"Session without target item found for user {last_uid}.")
            return
        if item_indices_session[0] == 0:
            item_indices_session = item_indices_session[1:]
        for subsequence_target_item_index in item_indices_session:
            subsequence_target_item_index += seq_start
            # Possible Subsequence from seq start to seq_start+subsequence_target_item_index - check the length
            subsequence_start_index = seq_start
            if subsequence_target_item_index - subsequence_start_index > max_item_list_len:
                subsequence_start_index = subsequence_target_item_index - max_item_list_len
            uid_list.append(last_uid)
            if masked_training:
                item_list_index.append(slice(subsequence_start_index + 1, subsequence_target_item_index + 1))
            else:
                item_list_index.append(slice(subsequence_start_index, subsequence_target_item_index))
            target_index.append(subsequence_target_item_index)
            item_list_length.append(subsequence_target_item_index - subsequence_start_index)

    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.

        """

        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        # For sequence features, join item attributes
        if self.item_feat is not None and self.iid_list_field in df:
            iid_list = [df[self.iid_list_field]]
            iid_feature_list = self.item_feat[iid_list]
            for k in iid_feature_list.interaction:
                df.interaction[k + "_list"] = iid_feature_list.interaction[k]
        return df


class GRU4RecAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class SASRecAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class NARMAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class CaserAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class COREAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class LightSANsAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class NextItNetAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)


class BERT4RecAttrDataset(BenchAttrSequentialDataset):
    def __init__(self, config):
        super().__init__(config)
