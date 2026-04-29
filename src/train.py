import pandas as pd
import importlib
from src.preprocess import Preprocessing_Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.evaluation import ModelEvaluation
from pydantic import BaseModel
from typing import Any, List, Dict




class DataConfig(BaseModel):
    interim_path: str
    train_path: str
    test_path: str

class TrainSettings(BaseModel):
    target_col: str
    test_size: float
    random_state: int

class PreprocessingConfig(BaseModel):
    numerical_cols: List[str]
    categorical_cols: List[str]
    service_cols: List[str]


class ModelConfig(BaseModel):
    module: str
    model_name: str
    params: Dict[str, Any]



class MetricConfig(BaseModel):
    module: str
    name: str


class AppConfig(BaseModel):
    paths: DataConfig
    train_settings: TrainSettings
    preprocessing: PreprocessingConfig
    model: ModelConfig
    metrics: List[MetricConfig]




class ModelTrain:
    def __init__(self, config: dict, logger):
        self.logger = logger


        try:
            self.config = AppConfig(**config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise


    def run_training(self):
        

        self.logger.info("Data is being loaded and separated into Train/Test sections...")
        df = pd.read_csv(self.config.paths.interim_path)

        target = self.config.train_settings.target_col
        X = df.drop(columns=[target])
        y = df[target].map({"Yes": 1, "No": 0})



        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.config.train_settings.test_size,
                                                            random_state=self.config.train_settings.random_state,
                                                            stratify=y
                                                            )


        self.logger.info("Preprocessing steps are being created...")
        preprocessing_pipeline = Preprocessing_Pipeline(logger=self.logger)


        preprocessing_steps = preprocessing_pipeline.create_pipeline(
            numerical_cols=self.config.preprocessing.numerical_cols,
            categorical_cols=self.config.preprocessing.categorical_cols,
            service_cols=self.config.preprocessing.service_cols
        )


        module_name = self.config.model.module
        model_name = self.config.model.model_name
        params = self.config.model.params

        self.logger.info(f"Loading model: {module_name}.{model_name}")
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, model_name) 
            model= model_class(**params)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Model library or class not found: {e}")
            raise

        full_model_pipeline = Pipeline(steps=[
            ("prep_steps", preprocessing_steps),
            ("model", model)
        ])



        self.logger.info("Pipeline training (Fit process)...")
        full_model_pipeline.fit(X_train, y_train)



        y_pred = full_model_pipeline.predict(X_test)

        if hasattr(full_model_pipeline, "predict_proba"):
            y_prob = full_model_pipeline.predict_proba(X_test)[:, 1] 
        else:
            y_prob = None


        self.logger.info("The Evaluation module is being called for model evaluation...")


        evaluator = ModelEvaluation(metrics_config=self.config.metrics, logger=self.logger)

        metrics_results = evaluator.evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)


        self.logger.info("Data is saved as train-test....")


        train_file = self.config.paths.train_path
        test_file = self.config.paths.test_path
      
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)


        return full_model_pipeline, metrics_results





        










