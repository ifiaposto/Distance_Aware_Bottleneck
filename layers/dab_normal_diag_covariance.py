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
from tensorflow.python.keras.utils import control_flow_util
import tensorflow as tf
import tensorflow_probability as tfp

from layers.dab_normal import _NormalDAB


class NormalDiagCovarianceDAB(_NormalDAB):
    
    """
    
    Dense layer with Distance-Aware Bottleneck.
    Diagonal covariance Normal distributions for both encoder and codebook entries are used.
    The distance from codebook is computed in terms KL divergence: KL(encoder,centroid).
 
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
            # number encoder's outputs (units): 
            #   dab_dim for the centroid means plus
            #   dab_dim for the centroid diagonal covariance matrices.
            units=dab_dim + dab_dim,
            dab_dim=dab_dim,
            codebook_size=codebook_size,
            dab_tau=dab_tau,
            momentum=momentum,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        
        super().build(input_shape)

        # The moving average for codebook's covariance matrices for supporting batch-manner updates 
        # of the Rate Distortion Finite Cardinality (RDFC) phase.    
        self.centroid_covariance_mavg = tf.Variable(
            name="centroid_covariance_mavg",
            shape=[self.codebook_size, self.dab_dim],
            initial_value=tf.zeros(shape=[self.codebook_size, self.dab_dim]),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype,
        )
        
        # Covariance matrix to be used in current inference step.
        # We initialize codebook's covariance matrices with the identity matrix (diagonal of 1s).
        self.centroid_covariance = tf.Variable(
            name="centroid_covariance",
            shape=[self.codebook_size, self.dab_dim],
            initial_value=tf.ones(shape=[self.codebook_size, self.dab_dim]),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype,
        )

        self.built = True

    def call(self, inputs, training=None):

        if training is None:
            training = tf.keras.backend.learning_phase()

        # encoders' params
        params = super()._call(inputs)

        mu, perturb_diag = tf.split(params, [self.dab_dim, self.dab_dim], axis=-1)


        perturb_diag = tf.math.softplus(perturb_diag - 5.0) + 1e-5

        # make covariance matrix psd.# Shift to keep small singular values and ease convergence.       
        encoder = tfp.distributions.MultivariateNormalDiag(
            loc=tf.expand_dims(mu, axis=1),
            scale_diag=tf.expand_dims(perturb_diag, axis=1),
        )

        # form distributions of centroids
        
        codebook_means = tf.expand_dims(self.centroid_means, axis=0)

        codebook_covariance = self.centroid_covariance

        # MultivariateNormalDiag accepts the scale S as an argument. Final covariance is C= S S^T
        codebook_covariance = tf.math.sqrt(tf.expand_dims(codebook_covariance, axis=0))
        
        
        codebook = tfp.distributions.MultivariateNormalDiag(
            loc=codebook_means, scale_diag=codebook_covariance
        )

        # distances_from_codebook [batch_size, codebook_size]
        # distances_from_codebook(i,j): distance of datapoint's encoder i from centroid j        
        distances_from_centroids = encoder.kl_divergence(codebook)
        
               
        # E-step: compute conditional assignment probabilities of datapoints to centroids
        # see equation p(h|x_i) on page 1731 in [1].
        log_centroid_probs = tf.reshape(
            tf.math.log(self.centroid_probs), shape=[1, self.codebook_size]
        )
        
        
        # the conditional assignment probabilities are prespecified.
        # Therefore, so we don't backpropagate during network's optimization.
        cond_centroid_probs = tf.nn.softmax(
            tf.stop_gradient(log_centroid_probs - self.dab_tau * distances_from_centroids),
            axis=-1,
        )

        # if first epoch, randomly assign a datapoint to centroid by uniform distribution
        cond_centroid_probs = tf.cond(
            self.initialized,
            lambda: cond_centroid_probs,
            lambda: tf.ones(shape=[tf.shape(inputs)[0], self.codebook_size])
            * 1
            / self.codebook_size,
        )


        # distance from codebook is the expected distance from centroids.
       
        # cond_centroid_probs: [batch_size,codebook_size]
        # distances_from_centroids: [batch_size,codebook_size]
        # distances_from_codebook: [batch_size]
        distances_from_codebook = tf.math.reduce_sum(
            cond_centroid_probs * distances_from_centroids, axis=-1
        )
        
        # M-step: compute moving averages

        # update moving average of prior centroid probabilities 
        self._update_centroid_probs(training, cond_centroid_probs)

        # update moving average of codebook's covariance matrices
        self._update_codebook_covariance(
            training, cond_centroid_probs, mu, perturb_diag * perturb_diag
        )

        # sample latent features
        outputs_sample = tf.squeeze(encoder.sample())

        # in eval mode, model is deterministic
        outputs = control_flow_util.smart_cond(
            training, lambda: outputs_sample, lambda: mu
        )

        # return learned features and distances from codebook (uncertainty)
        return tf.concat([outputs, tf.expand_dims(distances_from_codebook, axis=-1)], axis=-1)

    def set_codebook_covariance(self):

        """
        Utility function for setting the codebook's covariance matrices to their moving averages (the mavg copy) 
        for use in the next inference step. It is called at the end of the M-step of the
        Rate Distortion Finite Cardinality (RDFC).
        """

        self.centroid_covariance.assign(self.centroid_covariance_mavg)


    def _calculate_codebook_covariance(self, cond_centroid_probs, mu, covariance):

        """ "
        Utility function for computing the diagonal of the optimal codebook's covariance matrix of a centroid on current batch.
        Equation (9) in [1] is used, where off-diagonal entries are ignored. 

        References:

        [1] Davis J, Dhillon I. Differential entropic codebooking of multivariate gaussians. Advances in Neural Information Processing Systems. 2006;19.
        """

        replica_ctx = distribution_strategy_context.get_replica_context()
        batch_size = tf.cast(tf.shape(cond_centroid_probs)[0], tf.float32)
        
        
        # gather encoders' means and variances across all replicas
        if replica_ctx is not None:
            global_mu = replica_ctx.all_gather(mu, axis=0)
            global_covariance = replica_ctx.all_gather(covariance, axis=0)
            global_cond_centroid_probs = replica_ctx.all_gather(
                cond_centroid_probs, axis=0
            )
            global_batch_size = replica_ctx.all_reduce(
                reduce_util.ReduceOp.SUM, batch_size
            )
        else:
            global_mu = mu
            global_covariance = covariance
            global_cond_centroid_probs = cond_centroid_probs
            global_batch_size = batch_size

        # global mu: [batch_size, dab_dim] 
        # global_covariance: [batch_size, dab_dim] 
       

        # global_cond_centroid_probs: [batch_size, codebook_size]
        # global_cond_centroid_probs(i,j): how much datapoint i contributes to the covariance matrix of centroid j
        # we rectify datapoint's contribution for facilitating training.
        global_cond_centroid_probs = (global_cond_centroid_probs + 5.0) / (
            tf.math.reduce_sum(global_cond_centroid_probs, axis=0)+ 5 * global_batch_size)

        # global mu: [batch_size, 1, dab_dim] 
        # global_covariance: [batch_size, 1, dab_dim] 
        # centroid_means: [1, codebook_size, dab_dim] 
        global_mu = tf.expand_dims(global_mu, 1)
        global_covariance = tf.expand_dims(global_covariance, 1)
        centroid_means = tf.stop_gradient(tf.expand_dims(self.centroid_means, axis=0))
        

        # global_cond_centroid_probs: [batch_size, codebook_size, 1]
        global_cond_centroid_probs = tf.expand_dims(global_cond_centroid_probs, axis=-1)

        weighted_covariance = tf.math.multiply(global_cond_centroid_probs, global_covariance)

        weighted_covariance = tf.reduce_sum(weighted_covariance, axis=0)

        
        diff_means = global_mu - centroid_means

        diff_means_corr = diff_means * diff_means

        weighted_diff_means_corr = tf.math.multiply(global_cond_centroid_probs, diff_means_corr)

        weighted_diff_means_corr = tf.reduce_sum(weighted_diff_means_corr, axis=0)

        return weighted_covariance + weighted_diff_means_corr
