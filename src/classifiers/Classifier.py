"""
Abstract class for the NLP classification.
"""

from abc import ABC, abstractmethod

# from mlflow.pyfunc import PythonModel


class Classifier(ABC):
    """Abstract class that every Classifier should inherit from."""

    @abstractmethod
    def __init__(self):
        """Init method.

        Notes
        -----
        Model should be loaded in this method.
        """
        pass

    @abstractmethod
    def train(self):
        """Method to train the model."""
        raise NotImplementedError

    # @abstractmethod
    # def predict(self):
    #     """Method for the model to predict."""
    #     raise NotImplementedError
