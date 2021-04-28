from datetime import datetime


import numpy as np

import sklearn.metrics
import tensorflow as tf
from src.classifiers.Classifier import Classifier
from src.tools.viz import plot_confusion_matrix, plot_to_image


class KerasClassifier(Classifier):
    def __init__(self):
        pass

    def create_model(self):
        pass

    def preprocess_input(self, x):
        """Resizes the image.

        Parameters
        ----------
        x : nd.array
            Image

        Returns
        -------
        x : nd.array
            Image
        """
        pass

    def get_classification_head(self, x):
        """Classification head.

        Parameters
        ----------
        x :
            Tensor from previous layer.

        Returns
        prediction_layer :
            The layer with the probability for each of the 101 classes.
        """
        pass

    def train(
        self,
        train_dataset,
        validation_dataset,
        transfer_learning,
        n_epochs,
        learning_rate,
        fine_tuning=True,
        fine_tuning_epochs=20,
        fine_tuning_lr=0.00001,
    ):
        """Trains the model.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Training dataset.
        validation_dataset : tf.data.Dataset
            Validation dataset.
        n_epochs : int
            Number of epochs.
        learning_rate : float
            Learning rate.

        Returns
        -------
        history : keras.history
            The parameters like the loss that are saved during training.
        """

        now_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        logdir = "logs/" + now_date
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, profile_batch="100, 110"
        )
        file_writer_cm = tf.summary.create_file_writer(logdir + "/cm")

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=3
        )
        mc = tf.keras.callbacks.ModelCheckpoint(
            "best_model.h5",
            monitor="val_categorical_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True,
        )

        cc = CustomCallback(validation_dataset, file_writer_cm)

        if transfer_learning:
            # TODO
            # https://arxiv.org/pdf/1905.11946v5.pdf
            # tf.keras.optimizers.RMSprop(
            #     learning_rate=0.001,
            #     rho=0.9,
            #     momentum=0.0,
            #     epsilon=1e-07,
            #     centered=False,
            #     name="RMSprop",
            #     **kwargs
            # )


            # tf.keras.optimizers.Adam(lr=learning_rate)
            # lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=learning_rate, decay_steps=100)

            self.model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(
                #     learning_rate=0.001,
                #     rho=0.9,
                #     momentum=0.9,
                #     epsilon=1e-07,
                #     centered=False,
                #     name="RMSprop",
                # ),
                optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

            # Open the file
            with open('summary_transfer-learning.txt','w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

            print(train_dataset)
            history = self.model.fit(
                train_dataset,
                epochs=n_epochs,
                validation_data=validation_dataset,
                callbacks=[
                    tensorboard_callback,
                    es,
                    mc,
                    cc,
                ],
            )

        # logging.debug(history.history.keys())

        # https://keras.io/guides/transfer_learning/
        # https://www.tensorflow.org/tutorials/images/transfer_learning

        if fine_tuning:
            for layer in self.model.layers[: len(self.model.layers)]:
                layer.trainable = True


            self.model.compile(
                # optimizer=tf.keras.optimizers.RMSprop(
                #     learning_rate=0.001/10,
                #     rho=0.9,
                #     momentum=0.9,
                #     epsilon=1e-07,
                #     centered=False,
                #     name="RMSprop",
                # ),
                optimizer=tf.keras.optimizers.Adam(lr=fine_tuning_lr),
                # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

            # Open the file
            with open('summary_fine-tuning.txt','w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

            try: # if transfer_learning before
                history
                initial_epoch = history.epoch[-1]
                history = self.model.fit(
                    train_dataset,
                    initial_epoch=initial_epoch,
                    epochs=initial_epoch + fine_tuning_epochs + 1,
                    validation_data=validation_dataset,
                    callbacks=[tensorboard_callback, es, mc, cc, history],
                )
            except NameError:
                initial_epoch = 0
                history = self.model.fit(
                    train_dataset,
                    initial_epoch=initial_epoch,
                    epochs=initial_epoch + fine_tuning_epochs + 1,
                    validation_data=validation_dataset,
                    callbacks=[tensorboard_callback, es, mc, cc],
                )
                # history tf.keras.callbacks.History()


        return history, cc


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset, file_writer_cm):
        super().__init__()
        self.validation_data = validation_dataset
        self.file_writer_cm = file_writer_cm
        self.classification_reports = []
        self.cm_images = []

    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    # def on_epoch_end(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print(self.validation_data)
    #     batches = len(self.validation_data)
    #     batch_1, batch_2 = self.validation_data.take(2)
    #     images_batch_1, labels_batch_1 = batch_1
    #     print("images")
    #     print(images_batch_1)
    #     print("labels")
    #     print(labels_batch_1)
    #     preds_batch_1 = self.model.predict(images_batch_1)
    #     print("preds")
    #     print(preds_batch_1)
    #     print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        val_all_preds = []
        val_all_labels = []
        for i, val_batch_data in enumerate(self.validation_data):
            val_batch_imgs, val_batch_labels = val_batch_data
            val_batch_preds = self.model.predict(val_batch_data)
            val_all_preds.append(val_batch_preds)
            val_all_labels.append(val_batch_labels.numpy())
        val_all_preds = np.concatenate(val_all_preds, axis=0)
        val_all_labels = np.concatenate(val_all_labels, axis=0)

        class_names = get_class_names()

        cm = sklearn.metrics.confusion_matrix(
            y_true=np.argmax(val_all_labels, axis=1),
            y_pred=np.argmax(val_all_preds, axis=1),
            labels=[i for i in range(0, 101)],
        )
        figure = plot_confusion_matrix(cm, class_names=class_names)

        self.cm_images.append(figure)

        cm_image = plot_to_image(figure)
        figure.savefig("cm.png")
        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        classification_report = sklearn.metrics.classification_report(
            y_true=np.argmax(val_all_labels, axis=1),
            y_pred=np.argmax(val_all_preds, axis=1),
            labels=[i for i in range(0, 101)],
            zero_division=0,
            output_dict=True,
        )
        self.classification_reports.append(classification_report)


def get_class_names(file="src/tools/classes.txt"):
    with open(file, "r") as f:
        class_names = f.read().splitlines()
    return class_names
