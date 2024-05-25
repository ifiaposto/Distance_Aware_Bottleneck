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
    Wide Residual Network with Distance-Aware Bottleneck.

    This implementation is based on the implementation of Uncertainty Baselines:
    https://github.com/google/uncertainty-baselines/

"""


import functools
from typing import Iterable
from layers import NormalFullCovarianceDAB
import tensorflow as tf


BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9,
)


def Conv2D(filters, seed=None, **kwargs):  # pylint: disable=invalid-name
    """Conv2D layer that is deterministically initialized."""
    default_kwargs = {
        "kernel_size": 3,
        "padding": "same",
        "use_bias": False,
        # Note that we need to use the class constructor for the initializer to
        # get deterministic initialization.
        "kernel_initializer": tf.keras.initializers.HeNormal(seed=seed),
    }
    # Override defaults with the passed kwargs.
    default_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(filters, **default_kwargs)


def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int,
    version: int,
) -> tf.Tensor:
    """Basic residual block of two 3x3 convs.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      strides: Stride dimensions for Conv2D.
      conv_l2: L2 regularization coefficient for the conv kernels.
      bn_l2: L2 regularization coefficient for the batch norm layers.
      seed: random seed used for initialization.
      version: 1, indicating the original ordering from He et al. (2015); or 2,
        indicating the preactivation ordering from He et al. (2016).

    Returns:
      tf.Tensor.
    """
    x = inputs
    y = inputs
    if version == 2:
        y = BatchNormalization(
            beta_regularizer=tf.keras.regularizers.l2(bn_l2),
            gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
        )(y)
        y = tf.keras.layers.Activation("relu")(y)

    seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]

    y = Conv2D(
        filters,
        strides=strides,
        seed=seeds[0],
        kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
    )(y)

    y = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(bn_l2),
        gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
    )(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = Conv2D(
        filters,
        strides=1,
        seed=seeds[1],
        kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
    )(y)
    if version == 1:
        y = BatchNormalization(
            beta_regularizer=tf.keras.regularizers.l2(bn_l2),
            gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
        )(y)

    if not x.shape.is_compatible_with(y.shape):
        x = Conv2D(
            filters,
            kernel_size=1,
            strides=strides,
            seed=seeds[2],
            kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
        )(x)
    x = tf.keras.layers.add([x, y])
    if version == 1:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version, seed):
    """Group of residual blocks."""
    seeds = tf.random.experimental.stateless_split([seed, seed + 1], num_blocks)[:, 0]
    x = basic_block(
        inputs,
        filters=filters,
        strides=strides,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version,
        seed=seeds[0],
    )
    for i in range(num_blocks - 1):
        x = basic_block(
            x,
            filters=filters,
            strides=1,
            conv_l2=conv_l2,
            bn_l2=bn_l2,
            version=version,
            seed=seeds[i + 1],
        )
    return x


def wide_resnet_dab(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    dab_dim: int,
    dab_tau: float,
    codebook_size: int,
    version: int = 2,
    seed: int = 42,

) -> tf.keras.models.Model:
    """
     Builds Distance Aware Information Bottleneck Wide ResNet.

     Following Zagoruyko and Komodakis (2016), the backbone network accepts a width multiplier on the
     number of filters. Using three groups of residual blocks, the network maps
     spatial features of size 32x32 -> 16x16 -> 8x8.

     Args:
       input_shape: tf.Tensor.
       depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
         He et al. (2015)'s notation which uses the maximum depth of the network
         counting non-conv layers like dense.
       width_multiplier: Integer to multiply the number of typical filters by. "k"
         in WRN-n-k.
       num_classes: Number of output classes.
       l2: L2 regularization coefficient.
       version: 1, indicating the original ordering from He et al. (2015); or 2,
         indicating the preactivation ordering from He et al. (2016).
       seed: random seed used for initialization.
       dau_dim: Bottleneck dimension.
       dab_tau:Temperature of distances from codebook.
       codebook_size: Codebook size.


     Returns:
       tf.keras.Model.
    """
    
    seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
    if (depth - 4) % 6 != 0:
        raise ValueError("depth should be 6n+4 (e.g., 16, 22, 28, 40).")
    num_blocks = (depth - 4) // 6
    
    l2_reg = tf.keras.regularizers.l2
    
    # Build backbone network (encoder).
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = Conv2D(
        16,
        strides=1,
        seed=seeds[0],
        kernel_regularizer=l2_reg(l2),
    )(inputs)
    if version == 1:
        x = BatchNormalization(
            beta_regularizer=l2_reg(l2),
            gamma_regularizer=l2_reg(l2),
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
    x = group(
        x,
        filters=16 * width_multiplier,
        strides=1,
        num_blocks=num_blocks,
        conv_l2=l2,
        bn_l2=l2,
        version=version,
        seed=seeds[1],
    )
    x = group(
        x,
        filters=32 * width_multiplier,
        strides=2,
        num_blocks=num_blocks,
        conv_l2=l2,
        bn_l2=l2,
        version=version,
        seed=seeds[2],
    )
    x = group(
        x,
        filters=64 * width_multiplier,
        strides=2,
        num_blocks=num_blocks,
        conv_l2=l2,
        bn_l2=l2,
        version=version,
        seed=seeds[3],
    )

    if version == 2:
        x = BatchNormalization(
            beta_regularizer=l2_reg(l2),
            gamma_regularizer=l2_reg(l2),
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)

    # Build Distance-Aware Bottleneck.
    x = NormalFullCovarianceDAB(
        dab_dim=dab_dim,
        codebook_size=codebook_size,
        dab_tau=dab_tau,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
        kernel_regularizer=l2_reg(l2),
        bias_regularizer=l2_reg(l2),
        name="dense_dab",
    )(x)
    
    # Use x for prediction, d for distance of input from codebook (uncertainty).
    x, d = tf.split(x, [dab_dim, 1], axis=-1)

    # Build decoder.
    x = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
        kernel_regularizer=l2_reg(l2),
        bias_regularizer=l2_reg(l2),
    )(x)

    # Return predictions and corresponding uncertainty.
    x = tf.concat([x, d], axis=-1)

    return tf.keras.Model(
        inputs=inputs,
        outputs=x,
        name="wide_resnet_dab-{}-{}".format(depth, width_multiplier),
    )
