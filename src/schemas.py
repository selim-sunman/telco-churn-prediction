from pydantic import BaseModel
from typing import Any



class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    model_path: str
    metrics_path: str

class TrainSettings(BaseModel):
    target_col: str
    test_size: float
    random_state: int

class PreprocessingConfig(BaseModel):
    numerical_cols: list[str]
    categorical_cols: list[str]
    service_cols: list[str]


class ModelConfig(BaseModel):
    module: str
    model_name: str
    params: dict[str, Any]



class MetricConfig(BaseModel):
    module: str
    name: str


class AppConfig(BaseModel):
    paths: DataConfig
    train_settings: TrainSettings
    preprocessing: PreprocessingConfig
    model: ModelConfig
    metrics: list[MetricConfig]