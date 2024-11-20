# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""
import copy

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict

    def evaluate_sequence_lengths(self, dataobject: DataStruct, sequence_lengths):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        metrics_per_length = {}
        length_counts = {}
        for length in sequence_lengths:
            #filter dataobject by sequence length
            dataobject_filtered = copy.deepcopy(dataobject)
            seq_lenghts = dataobject_filtered._data_dict['rec.seq_len']
            #Select only the data with the desired sequence length
            mask = seq_lenghts == length
            dataobject_filtered._data_dict['rec.topk'] = dataobject_filtered._data_dict['rec.topk'][mask]
            dataobject_filtered._data_dict['rec.seq_len'] = dataobject_filtered._data_dict['rec.seq_len'][mask]
            result_dict = OrderedDict()
            if length_counts.get(length) == None:
                length_counts[length] = dataobject_filtered._data_dict['rec.topk'].shape[0]
            else:
                length_counts[length] += dataobject_filtered._data_dict['rec.topk'].shape[0]
            for metric in self.metrics:
                if dataobject_filtered._data_dict['rec.topk'].shape[0] != 0:
                    metric_val = self.metric_class[metric].calculate_metric(dataobject_filtered)
                    result_dict.update(metric_val)
            metrics_per_length[length] = result_dict
        return metrics_per_length, length_counts

    def evaluate_per_item(self, dataobject: DataStruct, item_ids):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        item_counts = {}
        metrics_per_length = OrderedDict()
        for item in item_ids:
            #filter dataobject by sequence length
            dataobject_filtered = copy.deepcopy(dataobject)
            pos_item = dataobject_filtered._data_dict['rec.pos_item']
            #Select only the data with the desired sequence length
            mask = pos_item == item
            dataobject_filtered._data_dict['rec.topk'] = dataobject_filtered._data_dict['rec.topk'][mask]
            dataobject_filtered._data_dict['rec.pos_item'] = dataobject_filtered._data_dict['rec.pos_item'][mask]
            result_dict = OrderedDict()
            if item_counts.get(item) == None:
                item_counts[item] = dataobject_filtered._data_dict['rec.topk'].shape[0]
            else:
                item_counts[item] += dataobject_filtered._data_dict['rec.topk'].shape[0]

            for metric in self.metrics:
                if dataobject_filtered._data_dict['rec.pos_item'].shape[0] != 0:
                    metric_val = self.metric_class[metric].calculate_metric(dataobject_filtered)
                    result_dict.update(metric_val)
            metrics_per_length[item] = result_dict
        return metrics_per_length, item_counts