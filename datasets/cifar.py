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
    CIFAR{10,100} dataset builders.
    
    This is a minimal implementation of the one created by the Uncertainty Baseline Authors:
        
    https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/datasets/cifar.py
    
"""

from typing import Optional, Union

import numpy as np
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from datasets import base


# We use the convention of using mean = np.mean(train_images, axis=(0,1,2))
# and std = np.std(train_images, axis=(0,1,2)).
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
# Previously we used std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
# which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype), however
# we change convention to use the std over the entire training set instead.


def load_corrupted_cifar_test_info():
    """Loads information for CIFAR-10-C."""

    corruption_types = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'frosted_glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic',
        'pixelate',
        'jpeg_compression',
    ]

    max_intensity = 5
    return corruption_types, max_intensity


def normalize_by_cifar(input_image, dtype=tf.float32, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    if input_image.dtype == tf.uint8:
        input_image = tf.image.convert_image_dtype(input_image, dtype)
    return (input_image - tf.constant(mean, dtype=dtype)) / tf.constant(
        std, dtype=dtype
    )


class _CifarDataset(base.BaseDataset):
    """CIFAR dataset builder abstract class."""

    def __init__(
        self,
        name: str,
        fingerprint_key: str,
        split: str,
        seed: Optional[Union[int, tf.Tensor]] = None,
        validation_percent: float = 0.0,
        shuffle_buffer_size: Optional[int] = None,
        num_parallel_parser_calls: int = 64,
        drop_remainder: bool = False,
        mask_and_pad: bool = False,
        normalize: bool = True,
        try_gcs: bool = False,
        download_data: bool = False,
        data_dir: Optional[str] = None,
        is_training: Optional[bool] = None,
    ):
        """Create a CIFAR10 or CIFAR100 tf.data.Dataset builder.

        Args:
          name: the name of this dataset, either 'cifar10', 'cifar100'.
          fingerprint_key: The name of the feature holding a string that will be
            used to create an element id using a fingerprinting function. If None,
            then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
            called and an `id` field is added to the example Dict.
          split: a dataset split, either a custom tfds.Split or one of the
            tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
            names.
          seed: the seed used as a source of randomness.
          validation_percent: the percent of the training set to use as a validation
            set.
          shuffle_buffer_size: the number of example to use in the shuffle buffer
            for tf.data.Dataset.shuffle().
          num_parallel_parser_calls: the number of parallel threads to use while
            preprocessing in tf.data.Dataset.map().
          drop_remainder: Whether or not to drop the last batch of data if the
            number of points is not exactly equal to the batch size.
          mask_and_pad: Whether or not to mask and pad batches such that when
            drop_remainder == False, partial batches are padded to a full batch and
            an additional `mask` feature is added to indicate which examples are
            padding.
          normalize: whether or not to normalize each image by the CIFAR dataset
            mean and stddev.
          try_gcs: Whether or not to try to use the GCS stored versions of dataset
            files.
          download_data: Whether or not to download data before loading.
          data_dir: Directory to read/write data, that is passed to the tfds
            dataset_builder as a data_dir parameter.
          is_training: Whether or not the given `split` is the training split. Only
            required when the passed split is not one of ['train', 'validation',
            'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
        """
        self._normalize = normalize

        dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
        if is_training is None:
            is_training = split in ["train", tfds.Split.TRAIN]
        new_split = base.get_validation_percent_split(
            dataset_builder, validation_percent, split
        )
        super().__init__(
            name=name,
            dataset_builder=dataset_builder,
            split=new_split,
            seed=seed,
            is_training=is_training,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_parser_calls=num_parallel_parser_calls,
            drop_remainder=drop_remainder,
            mask_and_pad=mask_and_pad,
            fingerprint_key=fingerprint_key,
            download_data=download_data,
            cache=True,
        )

    def _create_process_example_fn(self) -> base.PreProcessFn:
        def _example_parser(example: types.Features) -> types.Features:
            """A pre-process function to return images in [0, 1]."""
            image = example["image"]
            image_dtype = tf.float32

            if self._is_training:
                image_shape = tf.shape(image)
                # Expand the image by 2 pixels, then crop back down to 32x32.
                image = tf.image.resize_with_crop_or_pad(
                    image, image_shape[0] + 4, image_shape[1] + 4
                )
                # Note that self._seed will already be shape (2,), as is required for
                # stateless random ops, and so will per_example_step_seed.
                per_example_step_seed = tf.random.experimental.stateless_fold_in(
                    self._seed, example[self._enumerate_id_key]
                )
                # per_example_step_seeds will be of size (num, 3).
                # First for random_crop, second for flip, third optionally for
                # RandAugment, and foruth optionally for Augmix.
                per_example_step_seeds = tf.random.experimental.stateless_split(
                    per_example_step_seed, num=4
                )
                image = tf.image.stateless_random_crop(
                    image,
                    (image_shape[0], image_shape[0], 3),
                    seed=per_example_step_seeds[0],
                )
                image = tf.image.stateless_random_flip_left_right(
                    image, seed=per_example_step_seeds[1]
                )

   
            image = tf.image.convert_image_dtype(image, image_dtype)
            parsed_example = {"features": image}
            parsed_example[self._enumerate_id_key] = example[self._enumerate_id_key]
            if self._add_fingerprint_key:
                parsed_example[self._fingerprint_key] = example[self._fingerprint_key]

            parsed_example["labels"] = tf.cast(example["label"], tf.float32)

            return parsed_example

        return _example_parser


class Cifar10Dataset(_CifarDataset):
    """CIFAR10 dataset builder class."""

    def __init__(self, **kwargs):
        super().__init__(name="cifar10", fingerprint_key="id", **kwargs)


class Cifar100Dataset(_CifarDataset):
    """CIFAR100 dataset builder class."""

    def __init__(self, **kwargs):
        super().__init__(name="cifar100", fingerprint_key="id", **kwargs)


class Cifar10CorruptedDataset(_CifarDataset):
    """CIFAR10-C dataset builder class."""

    def __init__(self, corruption_type: str, severity: int, **kwargs):
        """Create a CIFAR10-C tf.data.Dataset builder.

        Args:
          corruption_type: Corruption name.
          severity: Corruption severity, an integer between 1 and 5.
          **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name=f"cifar10_corrupted/{corruption_type}_{severity}",
            fingerprint_key=None,
            **kwargs,
        )  # pytype: disable=wrong-arg-types  # kwargs-checking
