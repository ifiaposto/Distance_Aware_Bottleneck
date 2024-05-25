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

       Hyper-parameters and flags for training Distance-Aware Bottleneck ResNet-50 on ImageNet.

"""

from absl import flags

NUM_CLASSES = 1000
IMAGE_SHAPE = (224, 224, 3)

# Learning rate hyper-parameters.

flags.DEFINE_float('rdfc_learning_rate', 0.1,
                   'Base learning rate for the codebook.')
flags.DEFINE_float('base_learning_rate', 0.01,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('lr_decay_ratio', 0.1,
                   'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '65'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_integer('lr_warmup_epochs', 5,
                     'Number of epochs for a linear warmup to the initial learning rate. Use 0 to do no warmup.'
                     )
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')

# Train flags.

flags.DEFINE_integer('train_epochs', 70, 'Number of training epochs.')
flags.DEFINE_integer('per_core_batch_size', 256, 'Batch size per GPU.')
flags.DEFINE_integer('num_cores', 4, 'Number of GPUs.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

# Evaluation flags.

flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_list('ood_dataset', 'imageneto',
                  'List of OOD datasets to evaluate.')
flags.DEFINE_bool('eval_on_ood', True,
                  'Whether to run OOD evaluation on specified OOD datasets.'
                  )
flags.DEFINE_bool('eval_calibration', True,
                  "Whether to evaluate model's calibration.")

# Checkpoint flags.

flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to never save checkpoints.'
                     )
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')

# DAB hyper-parameters.

flags.DEFINE_float('beta', 0.01,
                   'Lagrange multiplier for DAB regularization.')
flags.DEFINE_float('dab_tau', 2.0, 'Temperature of DAB loss.')
flags.DEFINE_integer('dab_dim', 4, 'Bottleneck dimension.')
flags.DEFINE_integer('codebook_size', 1000, 'Codebook size. ')

# Train DAB hyper-parameters.

flags.DEFINE_bool('calibrate', True,
                  'Whether to encourage high uncertainty for the mispredicted training datapoints.'
                  )
flags.DEFINE_float('uncertainty_lb', 100.0,
                   'Uncertainty margin for improving calibration.')
flags.DEFINE_bool('backpropagate', True,
                  'Whether to backpropagate to the main network or not.'
                  )

FLAGS = flags.FLAGS
