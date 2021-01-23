"""This file configures the launch.
"""

from src.pipelines.mlflow_train_pipeline import train
from codecarbon import EmissionsTracker


tracking_uri = "http://127.0.0.1:5000"
exp_name = "efficientnetb4"


# tracker = EmissionsTracker()
# tracker.start()
# GPU Intensive code goes here
train(exp_name, tracking_uri)
# tracker.stop()
