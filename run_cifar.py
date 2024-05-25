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

    Train and Evaluate Wide ResNet 28-10 with Distance-Aware Bottleneck on CIFAR-10.

"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # print errors only

from absl import app
from absl import flags
from absl import logging

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import tensorflow_datasets as tfds

import datasets

from models import wide_resnet_dab
import utils.schedules as schedules
import utils.metrics_utils as metrics_utils
import hparams.cifar_hparams as cifar_hps

FLAGS = flags.FLAGS


def main(argv):
    
    # ===========================     experimental  setup        =========================== #

    ####################     set seeds      ####################

    tf.random.set_seed(FLAGS.seed)
    seeds = tf.random.experimental.stateless_split([FLAGS.seed, FLAGS.seed + 1], 2)[
        :, 0
    ]

    ####################     set-up directories      ####################

    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(
        dir_path,
        f"cifar_summaries_dab/seed_{FLAGS.seed}",
    )
    summary_writer = tf.summary.create_file_writer(output_dir)

    ##################     set-up distributed training      ##################

    strategy = tf.distribute.MirroredStrategy()
    batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

    ##################     set-up datasets                  ##################
    
    normalize_by_cifar=datasets.normalize_by_cifar

    train_builder = datasets.Cifar10Dataset(
        split=tfds.Split.TRAIN,
        seed=seeds[0],
        validation_percent=0.0,
    )
    train_dataset = train_builder.load(batch_size=batch_size,strategy=strategy)
   
    #

    test_builder = datasets.Cifar10Dataset(
        split=tfds.Split.TEST,
        drop_remainder=True,
    )
    
    test_dataset = test_builder.load(batch_size=batch_size,strategy=strategy)
    
    steps_per_epoch = train_builder.num_examples // batch_size
    steps_per_eval = test_builder.num_examples // batch_size

    ood_dataset_names = FLAGS.ood_dataset
    ood_datasets, steps_per_ood = datasets.load_ood_datasets(
        ood_dataset_names=ood_dataset_names,
        in_dataset_builder=test_builder,
        batch_size=batch_size,
        drop_remainder=True,
    )
    ood_datasets = {
                name: strategy.experimental_distribute_dataset(ds)
                for name, ds in ood_datasets.items()
                }

    if FLAGS.corruptions_interval > 0:
        corrupted_datasets = {}

        corruption_types, max_intensity = datasets.load_corrupted_cifar_test_info()

        for corruption_type in corruption_types:
            for severity in range(1, max_intensity + 1):
                dataset = datasets.Cifar10CorruptedDataset(
                    corruption_type=corruption_type,
                    severity=severity,
                    split=tfds.Split.TEST,
                    drop_remainder=True,
                ).load(batch_size=batch_size,strategy=strategy)
                corrupted_datasets[
                    f"{corruption_type}_{severity}"
                ] = dataset

    with strategy.scope():

        ##################     build model                  ##################
        NUM_CLASSES=cifar_hps.NUM_CLASSES

        logging.info("Building WideResNet model")
        
        model = wide_resnet_dab(
            input_shape=cifar_hps.IMAGE_SHAPE,
            depth=28,
            dab_dim=FLAGS.dab_dim,
            codebook_size=FLAGS.codebook_size,
            dab_tau=FLAGS.dab_tau,
            width_multiplier=10,
            num_classes=NUM_CLASSES,
            l2=FLAGS.l2,
            seed=seeds[1],
        )

        logging.info("Model input shape: %s", model.input_shape)
        logging.info("Model output shape: %s", model.output_shape)
        logging.info("Model number of weights: %s", model.count_params())
        model.summary()

        ##################     set-up optimizers    ##################
        # Linearly scale learning rate and the decay epochs by vanilla settings.
        base_lr = FLAGS.base_learning_rate * batch_size / 128

        train_epochs = FLAGS.train_epochs
        lr_decay_epochs = [
            (int(start_epoch_str) * train_epochs) // 200
            for start_epoch_str in FLAGS.lr_decay_epochs
        ]

        # encoder & decoder optimizer
        learning_rate = schedules.WarmUpPiecewiseConstantSchedule(
            steps_per_epoch=steps_per_epoch,
            base_learning_rate=base_lr,
            decay_ratio=FLAGS.lr_decay_ratio,
            decay_epochs=lr_decay_epochs,
            warmup_epochs=FLAGS.lr_warmup_epochs,
        )
        
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=1.0 - FLAGS.one_minus_momentum, 
            nesterov=True,
        )

        # codebook optimizer
        codebook_optimizer = tf.keras.optimizers.Adam(
            FLAGS.rdfc_learning_rate
        )


        ##################     set-up checkpoints    ##################
        
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=optimizer, codebook_optimizer=codebook_optimizer
        )

        latest_checkpoint = tf.train.latest_checkpoint(output_dir)
        initial_epoch = 0
        if latest_checkpoint:
            # checkpoint.restore must be within a strategy.scope() so that optimizer
            # slot variables are mirrored.
            checkpoint.restore(latest_checkpoint)
            logging.info("Loaded checkpoint %s", latest_checkpoint)
            initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

        if FLAGS.saved_model_dir:
            logging.info("Saved model dir : %s", FLAGS.saved_model_dir)
            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
            checkpoint.restore(latest_checkpoint)
            logging.info("Loaded checkpoint %s", latest_checkpoint)
        if FLAGS.eval_only:
            initial_epoch = train_epochs - 1  # Run just one epoch of eval

        ##################     create metrics    ##################

        metrics = {
            "train/negative_log_likelihood": tf.keras.metrics.Mean(),
            "train/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
            "train/loss": tf.keras.metrics.Mean(),
            "test/negative_log_likelihood": tf.keras.metrics.Mean(),
            "test/accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        }

        #OOD metrics.
        for dataset_name in ood_dataset_names:
            ood_metrics = metrics_utils.create_uncertainty_metrics(
                uncertainty_scores={"codebook_distance"},
                metric_prefix=dataset_name+"_ood",
                report_auprc=True, 
                report_auroc=True,
            )
            metrics.update(ood_metrics)
        
        #Calibration metrics. 
        calibration_metrics = metrics_utils.create_uncertainty_metrics(
            uncertainty_scores={"codebook_distance"},
            metric_prefix="calibration",
            report_auprc=False, 
            report_auroc=True,
        )
        metrics.update(calibration_metrics)

        #Noise corruption metrics.
        if FLAGS.corruptions_interval > 0:
            for dataset_name in corrupted_datasets:
                corrupted_ood_metrics = metrics_utils.create_uncertainty_metrics(
                    uncertainty_scores= {"codebook_distance"}, 
                    metric_prefix=dataset_name+"_corrupted",
                    report_auprc=True, 
                    report_auroc=True,
                )    
                metrics.update(corrupted_ood_metrics)
        

    # ===========================           learning algorithm        =========================== #

    @tf.function
    def rdfc_step(iterator):

        """
            This function computes Rate Distortion Finite Cardinality (RDFC) [1].
            In this phase, the codebook (the centroids) is trained.
            
            References
            
            [1] Banerjee A, Dhillon I, Ghosh J, Merugu S. An information theoretic analysis of 
                maximum likelihood mixture estimation for exponential families. ICML 2004.
            
        """

        def centroid_step_fn(inputs):
            """
                Train the codebook, i.e., find the centroids.
            """

            images = inputs["features"]

            with tf.GradientTape() as tape:
                
                images=normalize_by_cifar(images)
                
                outputs = model(images, training=True)

                _, uncertainty = tf.split(outputs, [NUM_CLASSES, 1], axis=-1)

                # Scale the loss given the distributed strategy will reduce sum all gradients.
                scaled_loss = (
                    tf.reduce_mean(uncertainty) / strategy.num_replicas_in_sync
                )

            # train only codebook 
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            codebook_optimizer.apply_gradients([(grad, var) for grad, var in zip(grads, model.trainable_variables) if "centroid"  in var.name])

        def centroid_probs_step_fn(inputs):
            """
                Gather conditional centroid probabilities (E-step) needed for the prior centroid probabilities.
            """
            
            images = inputs["features"]
            
            images=normalize_by_cifar(images)

            model(images, training=True)
            
        # =============================================================================

        # flag network as initialized
        model.get_layer("dense_dab").initialized.assign(
            tf.constant(True, dtype=tf.bool)
        )
        
        # update centroids
        model.get_layer("dense_dab").reset_codebook_covariance()
        for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
            strategy.run(centroid_step_fn, args=(next(iterator),))
        model.get_layer("dense_dab").set_codebook_covariance()

        # update centroid probabilities
        model.get_layer("dense_dab").reset_centroid_probs()
        for i in tf.range(tf.cast(steps_per_epoch, tf.int32)):
            strategy.run(centroid_probs_step_fn, args=(next(iterator),))
        model.get_layer("dense_dab").set_centroid_probs()

    @tf.function
    def train_step(iterator):
        """
            This step trains the encoder & decoder.
        """

        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs["features"]
            labels = inputs["labels"]

            with tf.GradientTape() as tape:
                images=normalize_by_cifar(images)
                outputs = model(images, training=True)

                logits, uncertainty = tf.split(outputs, [NUM_CLASSES, 1], axis=-1)

                negative_log_likelihood = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        labels, logits, from_logits=True
                    )
                )

                l2_loss = sum([l for l in model.losses])

                loss = (
                    negative_log_likelihood
                    + l2_loss
                    + FLAGS.beta * tf.reduce_mean(uncertainty)
                )
                # Scale the loss given the distributed strategy will reduce sum all gradients.
                scaled_loss = loss / strategy.num_replicas_in_sync

            grads = tape.gradient(scaled_loss, model.trainable_variables)

            # train only encoder/ decoder
            optimizer.apply_gradients([(grad, var) for grad, var in zip(grads, model.trainable_variables) if "centroid" not in var.name])

            metrics["train/loss"].update_state(loss)
            metrics["train/negative_log_likelihood"].update_state(
                negative_log_likelihood
            )
            metrics["train/accuracy"].update_state(labels, logits)
            
        # =============================================================================

        for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
            strategy.run(step_fn, args=(next(iterator),))

    #===========================           evaluation methods        =========================== #

    @tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
    def test_step(iterator, dataset_split, steps_per_eval):
        """
            Evaluation.
        """

        def step_fn(inputs):
            """Per-Replica StepFn."""
            images = inputs["features"]
            labels = inputs["labels"]
            images=normalize_by_cifar(images)
            outputs = model(images, training=False)

            logits, codebook_distance = tf.split(outputs, [NUM_CLASSES, 1], axis=-1)

            codebook_distance = tf.squeeze(codebook_distance)
                       
            ## prepare return values 
            return_values_dict = {}
            
            return_values_dict["codebook_distance"] = codebook_distance

            if dataset_split == "clean":
                metrics_utils.update_accuracy_metrics(metrics,labels,logits)            
                matches=tf.keras.metrics.sparse_categorical_accuracy(labels, logits)
                return_values_dict["matches"] = tf.cast(matches,tf.int32)        
            elif dataset_split == "ood":
                return_values_dict["ood_labels"] = 1 - inputs["is_in_distribution"]

            return return_values_dict

        # =============================================================================

        all_return_values_dict = {}

        # create arrays with 1 entry per batch evaluation
        if dataset_split == "clean":
            all_return_values_dict["matches"] = tf.TensorArray(
                tf.int32, size=steps_per_eval, dynamic_size=False
            )
        elif dataset_split == "ood":
            all_return_values_dict["ood_labels"] = tf.TensorArray(
                tf.int32, size=steps_per_eval, dynamic_size=False
            )
        all_return_values_dict["codebook_distance"] = tf.TensorArray(
            tf.float32, size=steps_per_eval, dynamic_size=False
        )

        for i in tf.range(tf.cast(steps_per_eval, tf.int32)):

            batch_return_values_dict = strategy.run(step_fn, args=(next(iterator),))
            

            for name, val in batch_return_values_dict.items():
                
                if not tf.is_tensor(batch_return_values_dict[name]):
                    batch_return_values_dict[name]=batch_return_values_dict[name].values

                # gather dicitionaries across replicas and update array
                all_return_values_dict[name] = all_return_values_dict[name].write(
                    i, tf.concat((batch_return_values_dict[name]), axis=0)
                )

        # return as tensor
        for name, val in all_return_values_dict.items():
            all_return_values_dict[name] = all_return_values_dict[name].concat()

        return all_return_values_dict

    
    
    # ===========================      training loop       =========================== #
    
    
    train_iterator = iter(train_dataset)

    for epoch in range(initial_epoch, train_epochs):

        if not FLAGS.eval_only:
            logging.info("Starting to train epoch: %s", epoch)
            
            # train encoder & decoder
            train_step(train_iterator)

            # train codebook to quantize datapoints' encoders
            logging.info("Starting RDFC epoch: %s", epoch)
            rdfc_step(train_iterator)

            logging.info(
                "Train Loss: %.4f, Test NLL: %.4f, Accuracy: %.2f%%",
                metrics["train/loss"].result(),
                metrics["train/negative_log_likelihood"].result(),
                metrics["train/accuracy"].result() * 100,
            )
            
            if (
                FLAGS.checkpoint_interval > 0
                and (epoch + 1) % FLAGS.checkpoint_interval == 0
            ):
                checkpoint_name = checkpoint.save(
                    os.path.join(output_dir, "checkpoint")
                )
                logging.info("Saved checkpoint to %s", checkpoint_name)

        

        logging.info("Starting to eval epoch: %s", epoch)
        
        test_iterator = iter(test_dataset)

        # evaluate on in-distribution, test data
        return_values = test_step(
                iterator=test_iterator, 
                dataset_split="clean", 
                steps_per_eval=steps_per_eval
        )

        logging.info(
            "Test NLL: %.4f, Accuracy: %.2f%%",
            metrics["test/negative_log_likelihood"].result(),
            metrics["test/accuracy"].result() * 100,
        )
        
        metrics_utils.update_uncertainty_metrics(
            strategy=strategy, 
            metrics=metrics, 
            metric_prefix="calibration",
            binary_labels=1-return_values["matches"], 
            uncertainty_scores=return_values["codebook_distance"],
            uncertainty_name= "codebook_distance"
        )

        # evaluate OOD detection
        for ood_dataset_name, ood_dataset in ood_datasets.items():

            ood_iterator = iter(ood_dataset)

            ood_return_values = test_step(
                ood_iterator,
                "ood",
                steps_per_ood[ood_dataset_name],
            )
            
            metrics_utils.update_uncertainty_metrics(
                strategy=strategy, 
                metrics=metrics, 
                metric_prefix=ood_dataset_name+"_ood",
                binary_labels=ood_return_values["ood_labels"], 
                uncertainty_scores=ood_return_values["codebook_distance"],
                uncertainty_name= "codebook_distance",
                update_auprc=True,
        )

        # evaluate noise-corrupted images detection
        if (
            FLAGS.corruptions_interval > 0
            and (epoch + 1) % FLAGS.corruptions_interval == 0
        ):

            for ood_dataset_name, ood_dataset in corrupted_datasets.items():

                ood_iterator = iter(ood_dataset)

                corrupted_return_values = test_step(
                    ood_iterator,
                    "corrupted",
                    steps_per_eval,
                )

               
                ood_scores= tf.concat(
                        [
                            return_values["codebook_distance"],
                            tf.squeeze(corrupted_return_values["codebook_distance"]),
                        ],
                        axis=-1,
                    )
                

                num_examples = return_values["codebook_distance"].shape[0]

                ood_labels = tf.concat(
                    [tf.zeros(shape=num_examples), tf.ones(shape=num_examples)], axis=0
                )

                metrics_utils.update_uncertainty_metrics(
                    strategy=strategy,
                    metrics=metrics,
                    metric_prefix=ood_dataset_name+"_corrupted",
                    binary_labels= ood_labels,
                    uncertainty_scores=ood_scores,
                    uncertainty_name="codebook_distance",
                    update_auprc=True,
                    verbose=False,
                )

        # update and reset metrics
        with summary_writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar(name, metric.result(), step=epoch + 1)

        for metric in metrics.values():
            metric.reset_states()
            
    # ===========================      save model and hyperparams       =========================== #

    checkpoint_name = checkpoint.save(os.path.join(output_dir, "checkpoint"))
    logging.info("Saved last checkpoint to %s", checkpoint_name)

    with summary_writer.as_default():
        hp.hparams(
            {
                "num_cores": FLAGS.num_cores,
                "base_learning_rate": FLAGS.base_learning_rate,
                "lr_decay_ratio":FLAGS.lr_decay_ratio,
                "rdfc_learning_rate": FLAGS.rdfc_learning_rate,
                "one_minus_momentum": FLAGS.one_minus_momentum,
                "l2": FLAGS.l2,
                "codebook_size": FLAGS.codebook_size,
                "bottleneck_dimension": FLAGS.dab_dim,
                "beta": FLAGS.beta,
                "temperature": FLAGS.dab_tau,
                "per_core_batch_size": FLAGS.per_core_batch_size,
                "seed": FLAGS.seed,            
            }
        )


if __name__ == "__main__":
    app.run(main)
