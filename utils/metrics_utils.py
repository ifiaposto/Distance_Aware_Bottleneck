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
    
    Utilities for reporting network's accuracy, calibration, and OOD detection.
    
"""

import tensorflow.compat.v2 as tf
import sklearn
from absl import logging

__all__ = [
    'update_accuracy_metrics',
    'create_uncertainty_metrics',
    'update_uncertainty_metrics',
]

@tf.function
def update_accuracy_metrics(
      metrics, labels, logits, metric_prefix='test', metric_suffix=''):
    
    """
        Compute and update accuracy metrics.
    """
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True))
    probs = tf.nn.softmax(logits)
    nll_key = metric_prefix + '/negative_log_likelihood' + metric_suffix
    metrics[nll_key].update_state(negative_log_likelihood)
    metrics[metric_prefix + '/accuracy' + metric_suffix].update_state(
        labels, probs)


def create_uncertainty_metrics(
        uncertainty_scores,metric_prefix="calibration",report_auprc=False, report_auroc=True):
    
    """
        Create binary classification metrics for evaluating uncertainty,i.e., 
        calibration, ood, or detection of 
        noise-corrupted inputs.
    """
    
    uncertainty_metrics = {}
    
    for score in uncertainty_scores:
        
        if report_auroc:
            uncertainty_metrics.update({f'{metric_prefix}_{score}_auroc':tf.keras.metrics.Mean()})
        
        if report_auprc:
            uncertainty_metrics.update({f'{metric_prefix}_{score}_auprc':tf.keras.metrics.Mean()})
            
    return uncertainty_metrics


def update_uncertainty_metrics(
        strategy, metrics, metric_prefix, binary_labels, uncertainty_scores, uncertainty_name, update_auroc=True, update_auprc=False, verbose=True):
    
    binary_labels=binary_labels.numpy()
    uncertainty_scores=uncertainty_scores.numpy()
    
    if update_auprc:
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(binary_labels, uncertainty_scores)
        auprc = sklearn.metrics.auc(x=recall, y=precision)
        if verbose:
            logging.info("Done with eval %s, %s AUPRC %.4f",metric_prefix,uncertainty_name, auprc,)


    if update_auroc:
        
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(binary_labels, uncertainty_scores)
        auroc = sklearn.metrics.auc(x=fpr, y=tpr)
        if verbose:
            logging.info("Done with eval %s, %s AUROC %.4f",metric_prefix, uncertainty_name, auroc,)
                
    @tf.function
    def update_uncertainty_metrics_fn():
        
        if update_auprc:
            metrics[f'{metric_prefix}_{uncertainty_name}_auprc'].update_state(auprc)
                    
        if update_auroc:                    
            metrics[f'{metric_prefix}_{uncertainty_name}_auroc'].update_state(auroc)
                    
                   
    strategy.run(update_uncertainty_metrics_fn)
    