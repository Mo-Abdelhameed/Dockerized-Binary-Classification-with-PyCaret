import warnings
import os
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from pycaret.classification import compare_models, setup, finalize_model, predict_model
from schema.data_schema import BinaryClassificationSchema

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = 'predictor.joblib'


class Classifier:
    """A wrapper class for the Random Forest binary classifier.

        This class provides a consistent interface that can be used with other
        classifier models.
    """

    model_name = 'pycaret_binary_classifier'

    def __init__(self, train_input: pd.DataFrame, schema: BinaryClassificationSchema):
        """Construct a new Binary Classifier."""
        self._is_trained = False
        self.schema = schema
        self.setup(train_input, schema)
        self.model = self.compare_models()

    def compare_models(self):
        """Build a new KNN binary classifier."""
        return compare_models()

    def setup(self, train_input: pd.DataFrame, schema: BinaryClassificationSchema):
        """Fit the KNN binary classifier to the training data.

        Args:
            train_input: The features of the training data.
            schema: The labels of the training data.
        """
        setup(train_input, target=schema.target, remove_outliers=True, normalize=True, ignore_features=[schema.id])
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the KNN binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the KNN binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the KNN binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        pipeline = finalize_model(self.model)
        joblib.dump(pipeline, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the KNN binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded KNN binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    @classmethod
    def train_predictor_model(cls, train_inputs: pd.DataFrame, train_targets: pd.Series,
                              hyperparameters: dict) -> "Classifier":
        """
        Instantiate and train the predictor model.

        Args:
            train_inputs (pd.DataFrame): The training data inputs.
            train_targets (pd.Series): The training data labels.
            hyperparameters (dict): Hyperparameters for the classifier.

        Returns:
            'Classifier': The classifier model
        """
        classifier = Classifier(**hyperparameters)
        classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
        return classifier

    @classmethod
    def predict_with_model(cls, classifier: "Classifier", data: pd.DataFrame, raw_score=False
                           ) -> pd.DataFrame:
        """
        Predict class probabilities for the given data.

        Args:
            classifier (Classifier): The classifier model.
            data (pd.DataFrame): The input data.
            raw_score (bool): Whether to return class probabilities or labels.
                Defaults to True.

        Returns:
            np.ndarray: The predicted classes or class probabilities.
        """
        return predict_model(classifier, data, raw_score=True)

    @classmethod
    def save_predictor_model(cls, model: "Classifier", predictor_dir_path: str) -> None:

        """
        Save the classifier model to disk.

        Args:
            model (Classifier): The classifier model to save.
            predictor_dir_path (str): Dir path to which to save the model.
        """
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)

    @classmethod
    def load_predictor_model(cls, predictor_dir_path: str) -> "Classifier":
        """
        Load the classifier model from disk.

        Args:
            predictor_dir_path (str): Dir path where model is saved.

        Returns:
            Classifier: A new instance of the loaded classifier model.
        """
        return Classifier.load(predictor_dir_path)

    @classmethod
    def evaluate_predictor_model(cls,
                                 model: "Classifier", x_test: pd.DataFrame, y_test: pd.Series
                                 ) -> float:
        """
        Evaluate the classifier model and return the accuracy.

        Args:
            model (Classifier): The classifier model.
            x_test (pd.DataFrame): The features of the test data.
            y_test (pd.Series): The labels of the test data.

        Returns:
            float: The accuracy of the classifier model.
        """
        return model.evaluate(x_test, y_test)
