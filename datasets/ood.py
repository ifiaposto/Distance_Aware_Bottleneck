# coding=utf-8
# Copyright 2024 Ifigeneia Apostolopoulou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Utilities for creating ood datasets.
    
    This is a minimal implementation based on the one created by the Uncertainty Baseline Authors:
        
    https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/datasets/base.py
    
    
"""

import logging
from typing import  Type, TypeVar,Callable, Sequence, Union

from robustness_metrics.common import ops
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf


from datasets.base import BaseDataset
from datasets import DATASETS


_BaseDatasetClass = Type[TypeVar("B", bound=BaseDataset)]

PreProcessFn = Callable[
    [Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]], types.Features
]


def make_ood_dataset(ood_dataset_cls: _BaseDatasetClass) -> _BaseDatasetClass:
    """Generate a BaseDataset with in/out distribution labels."""

    class _OodBaseDataset(ood_dataset_cls):
        """Combine two datasets to form one with in/out of distribution labels."""

        def __init__(
            self,
            in_distribution_dataset: BaseDataset,
            shuffle_datasets: bool = False,
            **kwargs
        ):

            super(_OodBaseDataset, self).__init__(**kwargs)
            # This should be the builder for whatever split will be considered
            # in-distribution (usually the test split).
            self._in_distribution_dataset = in_distribution_dataset
            self._shuffle_datasets = shuffle_datasets

        def load(self, *, batch_size: int = -1) -> tf.data.Dataset:
            # Set up the in-distribution dataset using the provided dataset builder.

            dataset_preprocess_fn = (
                self._in_distribution_dataset._create_process_example_fn()
            )  # pylint: disable=protected-access
            dataset_preprocess_fn = ops.compose(
                dataset_preprocess_fn, _create_ood_label_fn(True)
            )
            dataset = self._in_distribution_dataset.load(
                preprocess_fn=dataset_preprocess_fn, batch_size=batch_size
            )

            # Set up the OOD dataset using this class.

            ood_dataset_preprocess_fn = super(
                _OodBaseDataset, self
            )._create_process_example_fn()
            ood_dataset_preprocess_fn = ops.compose(
                ood_dataset_preprocess_fn, _create_ood_label_fn(False)
            )
            ood_dataset = super(_OodBaseDataset, self).load(
                preprocess_fn=ood_dataset_preprocess_fn, batch_size=batch_size
            )
            # We keep the fingerprint id in both dataset and ood_dataset

            # Combine the two datasets.
            try:
                combined_dataset = dataset.concatenate(ood_dataset)
            except TypeError:
                logging.info(
                    "Two datasets have different types, concat feature and label only"
                )

                def clean_keys(example):
                    # only keep features and labels, remove the rest
                    return {
                        "features": example["features"],
                        "labels": example["labels"],
                        "is_in_distribution": example["is_in_distribution"],
                    }

                combined_dataset = dataset.map(clean_keys).concatenate(
                    ood_dataset.map(clean_keys)
                )
            if self._shuffle_datasets:
                combined_dataset = combined_dataset.shuffle(self._shuffle_buffer_size)
            return combined_dataset

        def num_examples(self, data_type="default"):
            if data_type == "default":
                return self._in_distribution_dataset.num_examples + super().num_examples
            elif data_type == "in_distribution":
                return self._in_distribution_dataset.num_examples
            elif data_type == "ood":
                return super().num_examples
            else:
                raise NotImplementedError("The data_type %s is not valid." % data_type)

    return _OodBaseDataset


def _create_ood_label_fn(is_in_distribution: bool) -> PreProcessFn:
    """Returns a function that adds an `is_in_distribution` key to examles."""

    def _add_ood_label(example: types.Features) -> types.Features:
        if is_in_distribution:
            in_dist_label = tf.ones_like(example["labels"], tf.int32)
        else:
            in_dist_label = tf.zeros_like(example["labels"], tf.int32)
        example["is_in_distribution"] = in_dist_label
        return example

    return _add_ood_label


def load_ood_datasets(ood_dataset_names,
                      in_dataset_builder,
                      batch_size,
                      drop_remainder=False):
  """Load OOD datasets."""
  steps = {}
  datasets_dict = {}
  for ood_dataset_name in ood_dataset_names:
    ood_dataset_class = DATASETS[ood_dataset_name]
    ood_dataset_class = make_ood_dataset(ood_dataset_class)
    
    ood_dataset_builder = ood_dataset_class(
          in_dataset_builder,
          split='test',
          drop_remainder=drop_remainder,
          )
    
    ood_dataset = ood_dataset_builder.load(batch_size=batch_size)
    steps[f'{ood_dataset_name}'] = ood_dataset_builder.num_examples(
        'in_distribution') // batch_size + ood_dataset_builder.num_examples(
            'ood') // batch_size
    datasets_dict[f'{ood_dataset_name}'] = ood_dataset

  return datasets_dict, steps
