"""Training pipeline. Logs to MLflow.
"""

import argparse
import os
import tempfile

import mlflow
import mlflow.keras
import numpy as np
import plotly
import plotly.graph_objects as go
import tensorflow as tf
from src.classifiers.MobileNetV2 import InferMobileNetV2, MobileNetV2
from src.settings.settings import logging, mobilenetv2_params, xception_params
from src.tools.dataset import manual_get_datasets
import pandas as pd
from src.classifiers.Xception import Xception
from src.settings.settings import paths


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# tf.debugging.set_log_device_placement(True)

logging.debug(mobilenetv2_params)


def train(exp_name, tracking_uri):

    if exp_name == "mobilenetv2":
        params = mobilenetv2_params
    elif exp_name == "xception":
        params = xception_params

    data_path = paths["data"]["DATA_DIR"]
    train_dir = os.path.join(data_path, params["dataset"], "train_directory")
    test_dir = os.path.join(data_path,  params["dataset"], "test_directory")
    # mobilenetv2_weights = weights_path

    print(
        "Num GPUs Available: {}".format(
            len(tf.config.experimental.list_physical_devices("GPU"))
        )
    )
    print("List of GPUs {}".format(tf.config.list_physical_devices("GPU")))

    # tf.debugging.set_log_device_placement(True)
    # tf.config.set_soft_device_placement(True)

    print("Tracking uri : {}".format(tracking_uri))
    logging.debug("Tracking URI : {}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:

        logging.info("Run ID : {}".format(run.info.run_id))

        # All parameters are logged once thanks to mobilenetv2_params.yaml
        mlflow.log_params(params)
        # # For AzureML
        # mlflow.set_tags(mobilenetv2_params)

        train_dataset, validation_dataset = manual_get_datasets(train_dir, test_dir, params)
        # train_dataset, validation_dataset = auto_get_datasets(train_dir, test_dir, mobilenetv2_params)

        if params["test_mode"]:
            train_dataset = train_dataset.take(10)
            validation_dataset = validation_dataset.take(2)

        if exp_name == "mobilenetv2":
            # No need to change the parameters inside this script
            # Change the parameters in the mobilenetv2_params.yaml
            mobilenetv2_weights = os.path.join(paths["model"]["WEIGHTS_DIR"], params["weights"])
            model = MobileNetV2(
                mobilenetv2_weights,
                (
                    params["img_height"],
                    params["img_width"],
                    params["img_n_channels"],
                ),
            )
        elif exp_name == "xception":
            model = Xception((150, 150, 3))



        # history, cc = mobilenet.train(
        #     train_dataset,
        #     validation_dataset,
        #     n_epochs=mobilenetv2_params["n_epochs"],
        #     learning_rate=mobilenetv2_params["learning_rate"],
        #     fine_tuning=mobilenetv2_params["fine_tuning"],
        #     fine_tuning_epochs=mobilenetv2_params["fine_tuning_epochs"],
        # )
        history, cc = model.train(
            train_dataset,
            validation_dataset,
            n_epochs=params["n_epochs"],
            learning_rate=params["learning_rate"],
            fine_tuning=params["fine_tuning"],
            fine_tuning_epochs=params["fine_tuning_epochs"],
            fine_tuning_lr=params["fine_tuning_lr"]
        )

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        categorical_accuracy = history.history["categorical_accuracy"]
        val_categorical_accuracy = history.history["val_categorical_accuracy"]

        loss_fig = go.Figure()
        loss_fig.add_trace(
            go.Scatter(x=np.arange(mobilenetv2_params["n_epochs"]), y=loss, name="loss")
        )
        loss_fig.add_trace(
            go.Scatter(
                x=np.arange(mobilenetv2_params["n_epochs"]), y=val_loss, name="val_loss"
            )
        )

        metric_fig = go.Figure()
        metric_fig.add_trace(
            go.Scatter(
                x=np.arange(mobilenetv2_params["n_epochs"]),
                y=categorical_accuracy,
                name="categorical_accuracy",
            )
        )
        metric_fig.add_trace(
            go.Scatter(
                x=np.arange(mobilenetv2_params["n_epochs"]),
                y=val_categorical_accuracy,
                name="val_categorical_accuracy",
            )
        )

        tmpdir = tempfile.mkdtemp()
        loss_path = os.path.join(tmpdir, "loss.html")
        metric_path = os.path.join(tmpdir, "accuracy.html")
        plotly.offline.plot(loss_fig, filename=loss_path, auto_open=False)
        plotly.offline.plot(metric_fig, filename=metric_path, auto_open=False)
        logging.debug(tmpdir)
        mlflow.log_artifact(loss_path, artifact_path="graphs/loss.html")
        mlflow.log_artifact(metric_path, artifact_path="graphs/accuracy.html")

        # Save the best metric

        # Save best metric
        best_epoch = np.argmax(val_categorical_accuracy)
        mlflow.log_metric(
            "best_val_categorical_accuracy", max(val_categorical_accuracy)
        )
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("n_epochs", len(val_categorical_accuracy))

        # Save report
        report = cc.classification_reports[best_epoch]
        report_path = os.path.join(tmpdir, "report.csv")
        pd.DataFrame(report).transpose().to_csv(report_path)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Save confusion matrix
        cm = cc.cm_images[best_epoch]
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        cm.savefig(cm_path)
        mlflow.log_artifact(report_path, artifact_path="confusion_matrix.png")


        # Save the model

        # First log the Keras model thanks to MLFlow Keras API
        mlflow.keras.log_model(tf.keras.models.load_model("best_model.h5"), "model")

        # Then give the path to this artifact for the model to be used in inference
        artifacts = {
            "keras_model": "{}/model/data/model.h5".format(run.info.artifact_uri),
        }

        # Log the model to be used in Inference
        mlflow.pyfunc.log_model(
            artifact_path="py_model",
            python_model=InferMobileNetV2(),
            conda_env="environment.yml",
            artifacts=artifacts,
            code_path=["src/"],
        )

        # Clean up
        os.remove("cm.png")
        os.remove("best_model.h5")

    return run


if __name__ == "__main__":
    # retrieve the 2 arguments configured through `arguments` in the ScriptRunConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, dest="exp_name")
    parser.add_argument("--tracking-uri", type=str, dest="tracking_uri")
    parser.add_argument("--data-path", type=str, dest="data_path")
    parser.add_argument("--weights-path", type=str, dest="weights_path")

    args = parser.parse_args()
    exp_name = args.exp_name
    tracking_uri = args.tracking_uri
    data_path = args.data_path
    weights_path = args.weights_path

    train(exp_name, tracking_uri, data_path, weights_path)
