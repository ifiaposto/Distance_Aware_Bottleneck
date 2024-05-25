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

       Hyper-parameters and flags for training Distance-Aware Bottleneck Wide ResNet 28-10 on CIFAR-10.

"""

from absl import flags

NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)

# Learning rate hyper-parameters.

flags.DEFINE_float('rdfc_learning_rate', 0.1,
                   'Base learning rate for the centroids.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total batch size is 128. It is scaled by the ratio of the total batch size to 128.'
                   )
flags.DEFINE_float('lr_decay_ratio', 0.2,
                   'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial learning rate. Use 0 to do no warmup.'
                     )
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')

# Train flags.

flags.DEFINE_integer('train_epochs', 200,
                     'Number of training iterations.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per GPU.')
flags.DEFINE_integer('num_cores', 4, 'Number of GPUs.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Train hyper-parameters.

flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')

# Evaluation flags.

flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
                  'List of OOD datasets to evaluate on.')
flags.DEFINE_integer('corruptions_interval', 10,
                     'Number of epochs between evaluating on the corrupted test data. Use -1 to never evaluate.'
                     )

# Checkpoint flags.

flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to never save checkpoints.'
                     )
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')

# DAB hyper-parameters

flags.DEFINE_float('beta', 0.001,
                   'Lagrange multiplier for DAB regularization.')
flags.DEFINE_float('dab_tau', 1.0, 'Temperature of DAB loss.')
flags.DEFINE_integer('dab_dim', 8, 'Bottleneck dimension.')
flags.DEFINE_integer('codebook_size', 10, 'Codebook size. ')

FLAGS = flags.FLAGS
