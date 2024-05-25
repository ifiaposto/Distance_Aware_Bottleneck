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
    Pretrained  ResNet-50 with Distance-Aware Bottleneck.

"""


import tensorflow as tf
from layers import NormalDiagCovarianceDAB
from keras.applications.resnet import ResNet50


def pretrained_resnet50_dab(
    input_shape,
    num_classes,
    dab_dim,
    dab_tau,
    codebook_size,
    backpropagate=False,
):

    """Builds Distance Aware Information Bottleneck ResNet50 using pre-trained weights.

    Using strided conv, pooling, four groups of residual blocks, and pooling, the
    network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
    14x14 -> 7x7 (Table 1 of He et al. (2015)).

    Args:
      input_shape: Shape tuple of input excluding batch dimension.
      num_classes: Number of output classes.
      dau_dim: Bottleneck dimension.
      dab_tau:Temperature of distances from codebook.
      codebook_size: Codebook size.
      backpropagate: Flag indicating whether gradients will be backpropagated to the
                     backbone network (encoder).

      Returns:
          tf.keras.Model.

    """

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Build backbone network (encoder).
    bn = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        classes=num_classes,
    )

    # Designate whether encoder is trainable or not.
    bn.trainable = backpropagate

    features = bn(inputs)

    features = tf.keras.layers.GlobalAveragePooling2D()(features)

    features = tf.keras.layers.Flatten()(features)

    features = tf.cast(features, tf.float32)

    hidden_1 = tf.keras.layers.Dense(units=2048, activation=None, dtype=tf.float32)(
        features
    )

    hidden_1 = tf.keras.layers.Activation("relu", dtype=tf.float32)(hidden_1)

    hidden_3 = tf.keras.layers.Dense(units=2048, dtype=tf.float32)(hidden_1)

    hidden_3 = tf.keras.layers.Activation("relu", dtype=tf.float32)(hidden_3)

    hidden_4 = tf.keras.layers.Dense(units=2048, dtype=tf.float32)(hidden_3)

    hidden_4 = tf.keras.layers.Activation("relu", dtype=tf.float32)(hidden_4)

    # Build Distance-Aware Bottleneck.
    x = NormalDiagCovarianceDAB(
        dab_dim=dab_dim, codebook_size=codebook_size, dab_tau=dab_tau, name="dense_dab"
    )(hidden_4 + hidden_1)

    # Use x for prediction, d for distance of input from codebook (uncertainty).
    x, d = tf.split(x, [dab_dim, 1], axis=-1)

    # Build decoder.
    x = tf.keras.layers.Dense(
        num_classes, activation=None, dtype=tf.float32, name="decoder"
    )(x)

    # Return predictions and corresponding uncertainty.
    x = tf.concat([x, d], axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=x, name="resnet50_head")
