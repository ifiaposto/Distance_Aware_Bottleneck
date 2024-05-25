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

    A minimal library of Uncertainty Baselines training datasets.
    The full version can be found:

    https://github.com/google/uncertainty-baselines/tree/main/uncertainty_baselines/datasets
    
"""

# pylint: disable=g-bad-import-order

from datasets.base import BaseDataset
from datasets.cifar import normalize_by_cifar
from datasets.cifar import load_corrupted_cifar_test_info
from datasets.cifar import Cifar100Dataset
from datasets.cifar import Cifar10Dataset
from datasets.cifar import Cifar10CorruptedDataset
from datasets.imagenet import ImageNetDataset
from datasets.imagenet import ImageNetDatasetO
from datasets.svhn import SvhnDataset

DATASETS = {
    'cifar100': Cifar100Dataset,
    'cifar10': Cifar10Dataset,
    'cifar10_corrupted': Cifar10CorruptedDataset,
    'svhn_cropped': SvhnDataset,
    'imageneto': ImageNetDatasetO,
    'imagenet': ImageNetDataset,
}

from datasets.ood import make_ood_dataset
from datasets.ood import load_ood_datasets


__all__ = [
        "BaseDataset",
        "make_ood_dataset",
        "load_ood_datasets",
        "normalize_by_cifar",
        "load_corrupted_cifar_test_info",
        "Cifar100Dataset",
        "Cifar10Dataset",
        "Cifar10CorruptedDataset",
        "ImageNetDataset",
        "ImageNetDatasetO",
        "SvhnDataset",
        "DATASETS",
]


