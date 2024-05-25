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

    Piecewise constant learning rate schedule with warmup.
    
    This implementation is by Uncertainty Baselines Authors:
    
    https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/schedules.py

"""

import tensorflow as tf


class WarmUpPiecewiseConstantSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """Learning rate schedule.

    It starts with a linear warmup to the initial learning rate over
    `warmup_epochs`. This is found to be helpful for large batch size training
    (Goyal et al., 2018). The learning rate's value then uses the initial
    learning rate, and decays by a multiplier at the start of each epoch in
    `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
    """

    def __init__(
        self,
        steps_per_epoch,
        base_learning_rate,
        decay_ratio,
        decay_epochs,
        warmup_epochs,
    ):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.base_learning_rate = base_learning_rate
        self.decay_ratio = decay_ratio
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        learning_rate = self.base_learning_rate
        if self.warmup_epochs >= 1:
            learning_rate *= lr_epoch / self.warmup_epochs
        decay_epochs = [self.warmup_epochs] + self.decay_epochs
        for index, start_epoch in enumerate(decay_epochs):
            learning_rate = tf.where(
                lr_epoch >= start_epoch,
                self.base_learning_rate * self.decay_ratio**index,
                learning_rate,
            )
        return learning_rate

    def get_config(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "base_learning_rate": self.base_learning_rate,
            "decay_ratio": self.decay_ratio,
            "decay_epochs": self.decay_epochs,
            "warmup_epochs": self.warmup_epochs,
        }
