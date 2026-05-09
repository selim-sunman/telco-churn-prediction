import pandas as pd
import importlib
import joblib
import json
from src.preprocess import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.evaluation import ModelEvaluator
from src.schemas import AppConfig
from typing import Any
from pathlib import Path






class ModelTrainer:
    def __init__(self, config: dict, logger):
        self.logger = logger


        try:
            self.config = AppConfig(**config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise


    def setup_output_dirs(self):
         output_paths = [
             self.config.paths.model_path,
             self.config.paths.metrics_path            
         ]

         for path_str in output_paths:
             path = Path(path_str)

             path.parent.mkdir(parents=True, exist_ok=True)
             self.logger.info(f"Directory ensured: {path.parent}")


    def save_metrics(self, metrics: dict, path: Path):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Metrics were recorded successfully.: {path}")
        except Exception as e:
            self.logger.error(f"An error occurred while recording the metrics.: {e}")
            raise


    def run_training(self) -> tuple[Pipeline, dict[str, Any]]:

        self.setup_output_dirs()
        

        data_path = Path(self.config.paths.processed_path)
        self.logger.info("Data is being loaded and separated into Train/Test sections...")

        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError(f"The dataset read is empty: {data_path}")
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading data: {e}")
            raise

        target = self.config.train_settings.target_col
        X = df.drop(columns=[target])
        y = df[target].map({"Yes": 1, "No": 0})



        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.config.train_settings.test_size,
                                                            random_state=self.config.train_settings.random_state,
                                                            stratify=y
                                                            )


        self.logger.info("Preprocessing steps are being created...")
        preprocessing_pipeline = PreprocessingPipeline()


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


        try:
            self.logger.info("Pipeline training (Fit process)...")
            full_model_pipeline.fit(X_train, y_train)
        except ValueError as e:
            self.logger.error(f"Data error during model training (X or y may be mismatched): {e}")
        except Exception as e:
            self.logger.error(f"A technical problem occurred while training the model: {e}")



        y_pred = full_model_pipeline.predict(X_test)

        if hasattr(full_model_pipeline, "predict_proba"):
            y_prob = full_model_pipeline.predict_proba(X_test)[:, 1] 
        else:
            y_prob = None


        self.logger.info("The Evaluation module is being called for model evaluation...")


        evaluator = ModelEvaluator(metrics_config=self.config.metrics, logger=self.logger)

        metrics_results = evaluator.evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)

        


        metrics_save_path = Path(self.config.paths.metrics_path)
        self.logger.info(f"Metrics are being recorded: {metrics_save_path}")
        self.save_metrics(metrics=metrics_results, path=metrics_save_path)
        


        model_save_path = Path(self.config.paths.model_path)

        try:
            self.logger.info(f"Saving the model: {model_save_path}")
            joblib.dump(full_model_pipeline, model_save_path, compress=3)
        except OSError as e:
            self.logger.error(f"Disk error! Model could not be saved: {e}")
            raise
        

        return full_model_pipeline, metrics_results


        


        










