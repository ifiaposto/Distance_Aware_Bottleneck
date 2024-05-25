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

    ImageNet dataset builder.
    
    This is a minimal implementation of the one created by the Uncertainty Baseline Authors:
        
    https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/datasets/imagenet.py
    
    We have an option to use a percent of the training dataset as a validation set.
    We treat the original validation set as the test set. This is similar to what
    is also done in the NeurIPS uncertainty benchmark paper:
        
    https://arxiv.org/abs/1906.02530 (which used (100 / 1024)% as a validation set).
    
    Its also supports the ImageNet-O dataset:
        
    https://github.com/hendrycks/natural-adv-examples
    
"""

from typing import Dict, Optional, Union


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from datasets import base
from datasets import resnet_preprocessing


class _ImageNetDataset(base.BaseDataset):
    """ImageNet dataset builder abstract class."""

    def __init__(
        self,
        name: str,
        split: str,
        fingerprint_key: str,
        seed: Optional[Union[int, tf.Tensor]] = None,
        validation_percent: float = 0.0,
        shuffle_buffer_size: Optional[int] = 16384,
        num_parallel_parser_calls: int = 64,
        drop_remainder: bool = False,
        mask_and_pad: bool = False,
        try_gcs: bool = False,
        download_data: bool = False,
        data_dir: Optional[str] = None,
        load_data_dir: Optional[str] = None,
        is_training: Optional[bool] = None,
        image_size: int = 224,
        resnet_preprocessing_resize_method: Optional[str] = None,
        one_hot: bool = False,
    ):
        """Create an ImageNet tf.data.Dataset builder.

        Args:
          name: the name of this dataset.
          split: a dataset split, either a custom tfds.Split or one of the
            tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
            names.
         fingerprint_key: The name of the feature holding a string that will be
            used to create an element id using a fingerprinting function. If None,
            then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
            called and an `id` field is added to the example Dict.
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
          try_gcs: Whether or not to try to use the GCS stored versions of dataset
            files.
          download_data: Whether or not to download data before loading.
          data_dir: Directory to read/write data, that is passed to the tfds
            dataset_builder as a data_dir parameter.
          load_data_dir: Directory to find data if tfds.builder is built from directory.
          is_training: Whether or not the given `split` is the training split. Only
            required when the passed split is not one of ['train', 'validation',
            'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
          image_size: The size of the image in pixels.
          resnet_preprocessing_resize_method: Optional string for the resize method
            to use for resnet preprocessing.
          one_hot: whether or not to use one-hot labels.

        """

        if load_data_dir is None:
            dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
        else:
            dataset_builder = tfds.builder_from_directory(load_data_dir)

        if is_training is None:
            is_training = split in ["train", tfds.Split.TRAIN]

        if name == "imagenet2012":

            # use validation dataset as test set similar to the benchmarking paper:
            #  https://arxiv.org/abs/1906.02530
            new_split = base.get_validation_percent_split(
                dataset_builder,
                validation_percent,
                split,
                test_split=tfds.Split.VALIDATION,
            )
        else:
            new_split = base.get_validation_percent_split(
                dataset_builder,
                validation_percent,
                split,
            )

        super().__init__(
            name=name,
            dataset_builder=dataset_builder,
            split=new_split,
            is_training=is_training,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_parser_calls=num_parallel_parser_calls,
            drop_remainder=drop_remainder,
            mask_and_pad=mask_and_pad,
            fingerprint_key=fingerprint_key,
            download_data=download_data,
        )

        self._image_size = image_size
        self._resnet_preprocessing_resize_method = resnet_preprocessing_resize_method

        self._one_hot = one_hot

    def _create_process_example_fn(self) -> base.PreProcessFn:
        """Create a pre-process function to return images in [0, 1]."""

        def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            """Preprocesses ImageNet image Tensors."""
            per_example_step_seed = tf.random.experimental.stateless_fold_in(
                self._seed, example[self._enumerate_id_key]
            )

            image = example["image"]
            image = tf.cast(image, tf.float32)

            image = resnet_preprocessing.preprocess_image(
                image_bytes=example["image"],
                is_training=self._is_training,
                image_size=self._image_size,
                seed=per_example_step_seed,
                resize_method=self._resnet_preprocessing_resize_method,
            )

            if self._one_hot:
                label = tf.one_hot(example["label"], 1000, dtype=tf.float32)
            else:
                label = tf.cast(example["label"], tf.float32)
            parsed_example = {
                "features": image,
                "labels": label,
            }

            return parsed_example

        return _example_parser


class ImageNetDataset(_ImageNetDataset):
    """ImageNet dataset builder class."""

    # NOTE: Existing code passes in a split string as a positional argument, so
    # included here to preserve that behavior.
    def __init__(self, split, **kwargs):
        """Create an ImageNet tf.data.Dataset builder.

        Args:
          split: A dataset split, either a custom tfds.Split or one of the
            tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
            names.
          **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="imagenet2012", split=split, fingerprint_key="file_name", **kwargs
        )


class ImageNetDatasetO(_ImageNetDataset):
    """ImageNet dataset builder class."""

    # NOTE: Existing code passes in a split string as a positional argument, so
    # included here to preserve that behavior.
    def __init__(
        self, split, load_data_dir="~/tensorflow_datasets/imagenet_o/", **kwargs
    ):
        """Create an ImageNet tf.data.Dataset builder.

        Args:
          split: A dataset split, either a custom tfds.Split or one of the
            tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
            names.
          **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="imagenet_o",
            fingerprint_key=None,
            split=split,
            load_data_dir=load_data_dir,
            **kwargs
        )
