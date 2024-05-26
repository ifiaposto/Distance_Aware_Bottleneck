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


class NormalFullCovarianceDAB(_NormalDAB):
    
    """
    
    Dense layer with Distance-Aware Bottleneck.
    Full covariance Normal distributions for both encoder and codebook entries are used.
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
            #   (dab_dim * (dab_dim + 1)) // 2 for the centroid covariance matrices (note it is symmetric).
            units=dab_dim + (dab_dim * (dab_dim + 1)) // 2,
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
        # of the Rate Distortion Finite Cardinality phase.         
        self.centroid_covariance_mavg = tf.Variable(
            name="centroid_covariance_mavg",
            initial_value=tf.zeros(
                shape=[self.codebook_size, self.dab_dim, self.dab_dim],
            ),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype,
        )
   
        # We precompute the precision matrix and the log-determinant since these are the quantities
        # of codebook's covariance actually used for inference and evaluation.
        # For the moving average above, we kept the covariance due to its closed-form updates.
               
        # Precomputed codebook's precision matrices.
        self.centroid_precision = tf.Variable(
            name="centroid_precision",
            initial_value=tf.zeros(
                shape=[self.codebook_size, self.dab_dim, self.dab_dim],
            ),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype,
        )

        # We initialize codebook's precision matrices with the identity matrix.
        self.centroid_precision.assign(
            tf.tile(
                tf.expand_dims(tf.eye(self.dab_dim), axis=0), [self.codebook_size, 1, 1]
            )
        )
        
        # Precomputed log abs determinants of codebook's covariance matrices.Needed for the kl divergence.
        self.centroid_covariance_log_abs_det = tf.Variable(
            name="centroid_covariance_log_det",
            initial_value=tf.zeros(
                shape=[self.codebook_size, 1],
            ),
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

        mu, perturb_factor = tf.split(
            params, [self.dab_dim, (self.dab_dim * (self.dab_dim + 1)) // 2], axis=-1
        )

        # make covariance matrices symmetric and spd
        perturb_factor_low = tfp.math.fill_triangular(perturb_factor)
        perturb_factor_upper = tfp.math.fill_triangular(perturb_factor, upper=True)

        perturb_factor = 0.5 * (perturb_factor_upper + perturb_factor_low)

        # make covariance matrix psd
        perturb_diag, perturb_factor, _ = tf.linalg.svd(
            perturb_factor, full_matrices=True
        )
        # perturb factor is unitary, perturb_diag  contains non-negative entries
        
        # shift to keep small singular values and ease convergence
        perturb_diag = tf.math.softplus(perturb_diag - 5.0) + 1e-5
        
        
        # compute encoders' covariance matrices
        
        # The scale matrix is S=U L U^T
        # The covariance matrix is C=S S^T=UL U^T U L ^T=U L^2 U^T
        # U=perturb factor, L =diag(perturb_diag)
        
        covariance = tf.linalg.matmul(
            perturb_factor, tf.linalg.diag(perturb_diag * perturb_diag)
        )
        covariance = tf.linalg.matmul(covariance, perturb_factor, transpose_b=True)
           
        # encoder shape: [batch_size, 1, dab_dim]
        # we add a dimension to compute distances in vectorized form. 
        encoder = tfp.distributions.MultivariateNormalFullCovariance(
            loc=tf.expand_dims(mu, axis=1),
            covariance_matrix=tf.expand_dims(covariance, axis=1),
        )

        # distances_from_codebook [batch_size, codebook_size]
        # distances_from_codebook(i,j): distance of datapoint's encoder i from centroid j
        distances_from_centroids = self.compute_codebook_distance(encoder)
        
        # E-step: compute conditional assignment probabilities of datapoints to centroids
        # see equation p(h|x_i) on page 1731 in [1].
        log_centroid_probs = tf.reshape(
            tf.math.log(self.centroid_probs), shape=[1, self.codebook_size]
        )
        
        # the conditional assignment probabilities are prespecified.
        # Therefore, so we don't backpropagate during network's optimization.
        cond_centroid_probs = tf.nn.softmax(
            tf.stop_gradient(log_centroid_probs - self.dab_tau * distances_from_centroids), axis=-1
        )

        # if first epoch, randomly assign a datapoint to centroid by uniform distribution
        cond_centroid_probs = tf.cond(
            self.initialized,
            lambda: cond_centroid_probs,
            lambda: tf.ones(shape=[tf.shape(inputs)[0], self.codebook_size])
            * 1/ self.codebook_size,
        )
            
        # distance from codebook is the expected distance from centroids.
       
        # cond_centroid_probs: [batch_size,codebook_size]
        # distances_from_centroids: [batch_size,codebook_size]
        # distances_from_codebook: [batch_size]
        distances_from_codebook = tf.math.reduce_sum(cond_centroid_probs * distances_from_centroids, axis=-1)
    
        
        # M-step: compute moving averages
    
        # update moving average of prior centroid probabilities 
        self._update_centroid_probs(training, cond_centroid_probs)

        # update moving average of codebook's covariance matrices
        self._update_codebook_covariance(training, cond_centroid_probs, mu, covariance)

        # sample latent features
        outputs_sample = tf.squeeze(encoder.sample())

        # in eval mode, model is deterministic
        outputs = control_flow_util.smart_cond(
            training, lambda: outputs_sample, lambda: mu
        )

        # return learned features and distances from codebook (uncertainty)
        return tf.concat([outputs, tf.expand_dims(distances_from_codebook, axis=-1)], axis=-1)

       
    def compute_codebook_distance(self, encoder):
        
        """
            Compute expected distance in terms of KL(encoder,centroid) from codebook
            using the precomputed precision matrices and log-determinants of centroids.
        """

        batch_size = tf.shape(encoder.loc)[0]

        codebook_precision = tf.expand_dims(self.centroid_precision, axis=0)

        codebook_log_abs_det = tf.expand_dims(
            tf.reshape(self.centroid_covariance_log_abs_det, [self.codebook_size]),
            axis=0,
        )

        term_a = (
            -encoder.scale.log_abs_determinant()
            + 0.5 * codebook_log_abs_det
            + 0.5 * (-tf.cast(encoder.scale.domain_dimension_tensor(), encoder.dtype))
        )

        codebook_means = tf.stack(self.centroid_means)

        diff_mean = encoder.loc - codebook_means

        diff_2 = tf.expand_dims(diff_mean, axis=-1)

        c = tf.expand_dims(tf.linalg.matvec(codebook_precision, diff_mean), axis=-2)

        term_b = 0.5 * tf.linalg.matmul(c, diff_2, transpose_a=False)

        term_b = tf.reshape(term_b, [batch_size, self.codebook_size])

        term_c = 0.5 * tf.linalg.trace(
            tf.linalg.matmul(codebook_precision, encoder.covariance())
        )

        return term_a + term_b + term_c
    
                              
    def set_codebook_covariance(self):

        """
        Utility function for setting the codebook's covariance matrices to their moving averages (the mavg copy) 
        for use in the next inference step. It is called at the end of the M-step of the 
        Rate Distortion Finite Cardinality (RDFC) phase.
        """

        self.centroid_precision.assign(
            tf.linalg.inv(self.centroid_covariance_mavg)
        )
        
        log_det = tf.linalg.logdet(self.centroid_covariance_mavg)
        self.centroid_covariance_log_abs_det.assign(
            tf.expand_dims(log_det, axis=-1)
        )
       
        
    def _calculate_codebook_covariance(self, cond_centroid_probs, mu, covariance):

        """ "
            Utility function for computing the optimal codebook's covariance matrix of a centroid on current batch.
            Equation (9) in [1] is computed. 
    
            References:
    
            [1] Davis J, Dhillon I. Differential entropic codebooking of multivariate gaussians. 
            Advances in Neural Information Processing Systems. 2006;19.
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
        # global_covariance: [batch_size, dab_dim, dab_dim] 

        # global_cond_centroid_probs: [batch_size, codebook_size]
        # global_cond_centroid_probs(i,j): how much datapoint i contributes to the covariance matrix of centroid j
        # we consider a Dirichlet prior with ak=5.0 to avoid overconcentration and facilitating training.
        global_cond_centroid_probs = (global_cond_centroid_probs + 5.0) / (
            tf.math.reduce_sum(global_cond_centroid_probs, axis=0) + 5 * global_batch_size
        )
        
        # global mu: [batch_size, 1, dab_dim] 
        # global_covariance: [batch_size, 1, dab_dim, dab_dim] 
        # centroid_means: [1, codebook_size, dab_dim] 
        global_mu = tf.expand_dims(global_mu, 1)
        global_covariance = tf.expand_dims(global_covariance, 1)
        centroid_means = tf.stop_gradient(tf.expand_dims(self.centroid_means, axis=0))
        

        
        # diff_means: [batch_size, codebook, dab_dim] 
        diff_means = global_mu - centroid_means

        # diff_means: [batch_size, codebook, dab_dim, 1] 
        diff_means = tf.expand_dims(diff_means, axis=-1)


        diff_means_corr = tf.linalg.matmul(diff_means, diff_means, transpose_a=False, transpose_b=True)

        # the optimal covariance is the weighted average of encoders' covariances
        # plus rank one updates of the mean differences
        new_covariance = global_covariance + diff_means_corr


        # global_cond_centroid_probs: [batch_size, codebook_size, 1, 1]
        global_cond_centroid_probs = tf.expand_dims(global_cond_centroid_probs, axis=-1)
        global_cond_centroid_probs = tf.expand_dims(global_cond_centroid_probs, axis=-1)

        new_covariance = tf.math.multiply(global_cond_centroid_probs,new_covariance)
        new_covariance = tf.reduce_sum(new_covariance, axis=0)
       
        # [codebook_size,dab_dim,dab_dim]
        return new_covariance

