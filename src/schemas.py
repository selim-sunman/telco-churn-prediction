"""
schemas.py - Checks if config.yaml is written correctly.

This module uses Pydantic to make sure our settings are correct
before the program starts running. It checks things like "is test_size
a number?" so the program doesn't crash later on.
"""

from pydantic import BaseModel
from typing import Any


class DataConfig(BaseModel):
    """Holds the file paths for our data and models.

    Attributes:
        raw_path (str): Where the original data is stored.
        processed_path (str): Where to save the cleaned data.
        model_path (str): Where to save the trained model.
        metrics_path (str): Where to save the evaluation scores.
    """
    raw_path: str
    processed_path: str
    model_path: str
    metrics_path: str
    visualizer_path: str


class TrainSettings(BaseModel):
    """Holds settings for training the model.

    Attributes:
        target_col (str): The column we want to predict (e.g. "Churn").
        test_size (float): The percentage of data to keep for testing (e.g. 0.2).
        random_state (int): A number to make sure our random splits are the same every time.
    """
    target_col: str
    test_size: float
    random_state: int


class PreprocessingConfig(BaseModel):
    """Holds lists of columns for our preprocessing steps.

    Attributes:
        numerical_cols (list[str]): Columns with numbers to scale.
        categorical_cols (list[str]): Columns with text to encode.
        service_cols (list[str]): Columns used to count how many services a user has.
    """
    numerical_cols: list[str]
    categorical_cols: list[str]
    service_cols: list[str]


class ModelConfig(BaseModel):
    """Holds the settings to load the machine learning model.

    Attributes:
        module (str): The scikit-learn module (e.g. "sklearn.ensemble").
        model_name (str): The model class (e.g. "RandomForestClassifier").
        params (dict): The settings for the model (like max_depth).
    """
    module: str
    model_name: str
    params: dict[str, Any]


class Hyperparameter(BaseModel):
    module: str
    model_name: str
    params: dict[str, Any]


class MetricConfig(BaseModel):
    """Holds the settings for a single evaluation metric.

    Attributes:
        module (str): Where the metric is located (e.g. "sklearn.metrics").
        name (str): The name of the metric function (e.g. "f1_score").
    """
    module: str
    name: str


class AppConfig(BaseModel):
    """The main settings class that holds everything together.

    When we load config.yaml, it is passed into this class to make
    sure every section is correct.

    Attributes:
        paths: DataConfig settings.
        train_settings: TrainSettings settings.
        preprocessing: PreprocessingConfig settings.
        model: ModelConfig settings.
        metrics: A list of MetricConfig settings.
    """
    paths: DataConfig
    train_settings: TrainSettings
    preprocessing: PreprocessingConfig
    model: ModelConfig
    hyperparameters: Hyperparameter
    metrics: list[MetricConfig]