# -*- coding: utf-8 -*-

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

    Illustrate Distance-Aware Bottleneck on synthetic regression tasks.
    
"""

import numpy as np
from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
import edward2 as ed
import matplotlib.pyplot as plt

from layers import NormalFullCovarianceDAB

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 1500, 'Number of epochs.')
flags.DEFINE_integer('num_train_examples', 20,
                     'Number of training datapoints.')
flags.DEFINE_float('rdfc_learning_rate', 0.01,
                   'Learning rate for the codebook.')
flags.DEFINE_float('learning_rate', 0.001,
                   'Learning rate for main network.')

flags.DEFINE_float('beta', 1.0,
                   'Lagrange multiplier for DAB regularization.')
flags.DEFINE_float('dab_tau', 5.0, ' Temperature of DAB loss.')
flags.DEFINE_integer('dab_dim', 8, 'Bottleneck dimension')
flags.DEFINE_integer('example', 1,
                     'Example to be tested: 1. datapoints in the middle. 2: datapoints at the edges'
                     )
flags.DEFINE_integer('codebook_size', 1, 'Codebook size. ')

flags.DEFINE_bool('verbose', False, 'Print numerical details.')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_dataset():
    """
        Create dataset for the linear regression example.
    """

    if FLAGS.example == 1:
        x1 = np.random.uniform(-4, -0.0, size=FLAGS.num_train_examples
                               // 2).reshape((-1, 1))
        x2 = np.random.uniform(0.0, 4, size=FLAGS.num_train_examples
                               // 2).reshape((-1, 1))
    else:
        x1 = np.random.uniform(-5, -2.0, size=FLAGS.num_train_examples
                               // 2).reshape((-1, 1))
        x2 = np.random.uniform(2.0, 5, size=FLAGS.num_train_examples
                               // 2).reshape((-1, 1))

    x = np.concatenate([x1, x2])

    noise = np.random.normal(0, 9,
                             size=FLAGS.num_train_examples).reshape((-1,
            1))
    y = x ** 3 + noise

    x_ = np.linspace(-5, 5).reshape((-1, 1))
    y_ = x_ ** 3

    x = np.float32(x)
    y = np.float32(y)

    x_ = np.float32(x_)
    y_ = np.float32(y_)

    return (y, x, x_, y_)


def multilayer_perceptron():
    """
        Create model.
    """

    inputs = tf.keras.layers.Input(shape=1)

    hidden_1 = tf.keras.layers.Dense(units=100, activation='elu'
            )(inputs)

    hidden_2 = tf.keras.layers.Dense(units=100, activation='elu'
            )(hidden_1)

    dab_output = NormalFullCovarianceDAB(
        dab_dim=FLAGS.dab_dim,
        codebook_size=FLAGS.codebook_size,
        dab_tau=FLAGS.dab_tau,
        name='dense_dab',
        activation=None,
        momentum=0.0,
        )(hidden_2)

    (latent_features, uncertainty) = tf.split(dab_output,
            [FLAGS.dab_dim, 1], axis=-1)

    output = tf.keras.layers.Dense(units=1,
                                   activation=None)(latent_features)

    return tf.keras.Model(inputs=inputs, outputs=tf.concat([output,
                          uncertainty], axis=-1))


def main(argv):

    (y_train, x_train, x_test, y_test) = create_dataset()

    model = multilayer_perceptron()

    codebook_optimizer = \
        tf.keras.optimizers.Adam(FLAGS.rdfc_learning_rate)

    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)

    @tf.function
    def rdfc_step(inputs):
        """
            This function computes Rate Distortion Finite Cardinality (RDFC) [1].
            In this phase, the codebook (the centroids) is trained.
            
            References
            
            [1] Banerjee A, Dhillon I, Ghosh J, Merugu S. An information theoretic analysis of 
                maximum likelihood mixture estimation for exponential families. ICML 2004.
        """

        def centroid_step_fn(inputs):
            """
            Train the codebook.
            """

            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)

                (_, uncertainty) = tf.split(outputs, 2, axis=-1)

                loss = tf.reduce_mean(uncertainty)

                grads = tape.gradient(loss, model.trainable_variables)

                # train centroids

                grads_and_vars = []
                for (grad, var) in zip(grads,
                        model.trainable_variables):
                    if 'centroid' in var.name:
                        grads_and_vars.append((grad, var))
                codebook_optimizer.apply_gradients(grads_and_vars)

        def centroid_probs_step_fn(inputs):
            """
                Gather conditional centroid probabilities (E-step) needed for 
                the prior centroid probabilities.
            """

            model(inputs, training=True)

        # flag network as initialized

        model.get_layer('dense_dab'
                        ).initialized.assign(tf.constant(True,
                dtype=tf.bool))

        # update centroids

        model.get_layer('dense_dab').reset_codebook_covariance()
        centroid_step_fn(inputs)
        model.get_layer('dense_dab').set_codebook_covariance()

        # update prior centroid probabilities

        model.get_layer('dense_dab').reset_centroid_probs()
        centroid_probs_step_fn(inputs)
        model.get_layer('dense_dab').set_centroid_probs()

    @tf.function
    def train_step(inputs, labels):
        """
            This step trains the encoder & decoder.
        """

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)

            (predictions, uncertainty) = tf.split(outputs, 2, axis=-1)

            distribution = tf.keras.layers.Lambda(lambda x: \
                    ed.Normal(loc=x, scale=1.0))(predictions)

            negative_log_likelihood = \
                -distribution.distribution.log_prob(labels)

            l2_loss = sum([l for l in model.losses])

            loss = negative_log_likelihood + l2_loss + FLAGS.beta \
                * tf.reduce_mean(uncertainty)

        grads = tape.gradient(loss, model.trainable_variables)

        # train encoder &  decoder

        grads_and_vars = []
        for (grad, var) in zip(grads, model.trainable_variables):
            if 'centroid' not in var.name:
                grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)

    # ===========================      training loop       =========================== #

    for epoch in range(FLAGS.epochs):

        if FLAGS.verbose:
            logging.info('Starting to train epoch: %s', epoch)

        train_step(x_train, y_train)

        rdfc_step(x_train)

    # ===========================      qualitative evaluation       =========================== #

    output = model.predict(x_test)
    (y_pred, uncertainty) = tf.split(output, 2, axis=-1)

    y_pred_low = np.squeeze(y_pred - 2 * uncertainty)
    y_pred_high = np.squeeze(y_pred + 2 * uncertainty)

    plt.fill_between(
        np.squeeze(x_test),
        y_pred_low,
        y_pred_high,
        color='coral',
        alpha=0.5,
        label='Uncertainty',
        )

    plt.plot(x_test, y_pred, c='royalblue', label='Prediction',
             linewidth=2)

    plt.scatter(x_train, y_train, c='navy', label='Train Datapoint')
    plt.plot(x_test, y_test, c='grey', label='Ground Truth',
             linewidth=2)

    if FLAGS.example == 1:
        plt.legend(loc='upper center')
        plt.legend().set_visible(True)
    else:
        plt.legend().set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app.run(main)
