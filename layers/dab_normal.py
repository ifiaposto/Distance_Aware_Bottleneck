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

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util
import tensorflow as tf


class _NormalDAB(tf.keras.layers.Dense):

    """
    Abstract class for a dense layer with Distance-Aware Bottleneck.
    Normal distributions for both encoder and codebook entries are used.
 
    Args:
         -------------------------------------------------------------------------------------------------------
         Codebook-related Args:
             
         units: Number of encoder's params (means + covariances).
         dab_dim: Dimension of latent features.
         codebook_size: The codebook size.
         dab_tau: Temperature in distance's from the codebook.
         momentum: Momentum for the moving average.
             Moving average is used for a batch-manner update of codebook's covariance matrices and
             prior probabilities.
         activation: Non-linearity to be applied to the input features before they are passed to
             encoder's last layer.
         -------------------------------------------------------------------------------------------------------
         Dense layer-related Args:
             
         use_bias: Boolean, whether the layer uses a bias vector.
         kernel_initializer: Initializer for the `kernel` weights matrix.
         bias_initializer: Initializer for the bias vector.
         kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
         bias_regularizer: Regularizer function applied to the bias vector.
         activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
         kernel_constraint: Constraint function applied to the `kernel` weights matrix.
         bias_constraint: Constraint function applied to the bias vector.


     Output:

         Output shape: [None, dab_dim+1].

         outputs[: dam_dim]: latent features.
         outputs[dam_dim:]: distance from codebook (uncertainty).

     References:

       [1] Alemi AA, Fischer I, Dillon JV, Murphy K. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410. 2016 Dec 1.
       [2] Alemi AA, Fischer I, Dillon JV. Uncertainty in the variational information bottleneck. arXiv preprint arXiv:1807.00906. 2018 Jul 2.

    """

    def __init__(
        self,
        units,
        dab_dim,
        codebook_size,
        dab_tau=1.0,
        momentum=0.99,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):

        super().__init__(
            units=units,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        # DAB hyperparams
        self.dab_activation = tf.keras.activations.get(activation)
        self.dab_dim = dab_dim
        self.codebook_size = codebook_size
        self.dab_tau = dab_tau
        self.momentum = momentum

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        super().build(input_shape)

        # centroid means
        self.centroid_means = super().add_weight(
            name="centroid_means",
            shape=[self.codebook_size, self.dab_dim],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
            trainable=True,
            dtype=self.dtype,
        )

        # prior centroid propabilities (maintain two copies)
        # the prior centroid probabilities that is used in current epoch.
        self.centroid_probs = tf.Variable(
            initial_value=tf.ones(shape=[self.codebook_size]) * 1 / self.codebook_size,
            trainable=False,
            name="centroid_probabilities",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=[self.codebook_size],
        )
        # the prior centroid probabilities moving average updated in batch-mode for use in the next epoch.
        self.centroid_probs_mavg = tf.Variable(
            initial_value=tf.ones(shape=[self.codebook_size]) * 1 / self.codebook_size,
            trainable=False,
            name="centroid_probabilities_mavg",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            shape=[self.codebook_size],
        )
        
    
        # Flag needed for random assignment of datapoints to centroids in the first epoch.
        self.initialized = self.add_weight(
            name="init",
            dtype=tf.bool,
            shape=(),
            initializer="zeros",
            trainable=False,
        )

        self.built = True

    def _call(self, inputs, training=None):

        """
            Compute encoder parameters.
        """

        inputs = self.dab_activation(inputs)

        return super().call(inputs)

    def call(self, inputs, training=None):

        """
            Main method for getting DAB's latent features and distances.
        """

        raise NotImplementedError(
            f"Normal DAB '{self.__class__.__name__}' must override `call(...)`."
        )

    # ===========================      Utility functions for centroid probabilities      =========================== #

    def _update_centroid_probs(self, training, cond_centroid_probs):

        """
             Utility function for updating the moving average of prior centroid probabilities (mavg copy) with estimates of current batch,
             if in training phase.
        """


        if training is None:
            training = tf.keras.backend.learning_phase()
            
        if training:
            new_centroid_probs = self._calculate_centroid_probs(cond_centroid_probs)
            self.centroid_probs_mavg.assign(
                self.centroid_probs_mavg * self.momentum
                + new_centroid_probs * (1.0 - self.momentum)
            )

    def reset_centroid_probs(self):
        """ 
            Utility function for reseting the moving average prior of centroid probabilities (the mavg copy) to 
            uniform distribution at the begining of a new epoch. It is called at the beginning of the M-step of 
            the Rate Distortion Finite Cardinality (RDFC) phase.
        """
        self.centroid_probs_mavg.assign(
            tf.ones(shape=[self.codebook_size]) * 1 / self.codebook_size
        )

    def set_centroid_probs(self):

        """
            Utility function for setting the prior centroid probabilities to their moving averages (the mavg copy) 
            for use in the next epoch. It is called at the end of the M-step of the 
            Rate Distortion Finite Cardinality (RDFC) phase.
        """

        self.centroid_probs.assign(self.centroid_probs_mavg)

        return

    def _calculate_centroid_probs(self, cond_centroid_probs):

        """
            Utility function for calculating prior centroid probalities from conditional centroid probabilities of current batch.
            Specifically, it computes Equation of \pi(h) with equal datapoint weights vi (page 1731) in [1].
    
            References:
    
            [1] Banerjee A, Merugu S, Dhillon IS, Ghosh J, Lafferty J. Clustering with Bregman divergences. 
                Journal of machine learning research. 2005 Oct 1;6(10).

        """
        
        
        replica_ctx = distribution_strategy_context.get_replica_context()
        batch_size = tf.cast(tf.shape(cond_centroid_probs)[0], tf.float32)
        
        # Single-GPU training.
        if replica_ctx is None:
            
            centroid_probs = tf.nn.relu(
                tf.reduce_sum(cond_centroid_probs, axis=0, keepdims=False) / batch_size
            )
            return centroid_probs
        
        # Multi-GPU training.
        
        # cond_centroid_probs: [batch_size, codebook_size]
        # local sum: [codebook_size]
        local_sum = tf.reduce_sum(cond_centroid_probs, axis=0, keepdims=False)
            
        # gather sums conditional probabilities from all replicas
        global_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, local_sum)

        # get number of training datapoints across all replicas
        global_batch_size = replica_ctx.all_reduce(
                reduce_util.ReduceOp.SUM, batch_size
            )
            
        # Apply relu in case floating point rounding causes it to go negative.
        centroid_probs = tf.nn.relu(global_sum / global_batch_size)
            
        # centroid_probs: [codebook_size]
        
        return centroid_probs
  

    # ===========================      Utility functions for codebook's covariance matrices     =========================== #

    def reset_codebook_covariance(self):
        """
            Utility function for reseting the moving average of codebook's covariance matrices (the mavg copy) to
            zero at the begining of the inference step. It is called at the beginning of the M-step of the 
            Rate Distortion Finite Cardinality (RDFC) phase.
        """

        self.centroid_covariance_mavg.assign(
            tf.zeros(shape=self.centroid_covariance_mavg.shape)
        )

    def set_codebook_covariance(self):

        """
            Utility function for setting the covariance matrices to their moving averages (the mavg copy) 
            for use in the next inference step. It is called at the end of the M-step of the 
            Rate Distortion Finite Cardinality (RDFC) phase.
        """

        raise NotImplementedError(
            f"Normal DAB '{self.__class__.__name__}' must override `set_codebook_covariance(...)`."
        )

    def _update_codebook_covariance(self, training, cond_centroid_probs, mu, covariance):

        """
            Utility function for updating the moving average of codebook's covariances (mavg copy) with 
            estimates of current batch, if in training phase.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            new_centroid_covariance = self._calculate_codebook_covariance(
                cond_centroid_probs, mu, covariance
            )

            self.centroid_covariance_mavg.assign(
                self.centroid_covariance_mavg * self.momentum
                + new_centroid_covariance * (1.0 - self.momentum)
            )

    def _calculate_codebook_covariance(self, cond_centroid_probs, mu, covariance):
        """
            Utility function for calculating codebook's covariance matrices from conditional centroid probabilities
            and encoders' means and covariance of current batch.
        """

        raise NotImplementedError(
            f"Normal DAB '{self.__class__.__name__}' must override `calculate_codebook_covariance(...)`."
        )
        

    def get_config(self):
        """
            Needed for DAB's serialization.
        """
        config = {
            "dab_activation": tf.keras.activations.serialize(self.dab_activation),
            "dab_dim": self.dab_dim,
            "codebook_size": self.codebook_size,
            "dab_tau": self.dab_tau,
            "momentum": self.momentum,
        }
        new_config = super().get_config()
        new_config.update(config)
        return new_config
