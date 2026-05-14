"""
train.py - Trains the machine learning model.

This module handles the entire training process:
- Loads the cleaned dataset
- Splits it into training and test sets
- Builds the preprocessing + model pipeline
- Trains the model
- Evaluates performance on the test set
- Saves the model and metrics to disk

The model and its settings are read from config.yaml,
so you can change the algorithm without touching this file.
"""

import pandas as pd
import importlib
import joblib
import json
from src.preprocess import PreprocessingPipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.evaluation import ModelEvaluator
from src.schemas import AppConfig
from src.visualizer import ModelVisualizer
from typing import Any
from pathlib import Path


class ModelTrainer:
    """Manages the full model training workflow.

    Reads the processed dataset, builds a pipeline that combines
    preprocessing and the classifier, trains it, evaluates it,
    and saves the results to disk.

    The model is loaded dynamically from config.yaml — you can switch
    between LogisticRegression, RandomForest, etc. with just a YAML edit.

    Attributes:
        config (AppConfig): Project settings loaded from config.yaml.
        logger: Logger used to print progress and error messages.
    """

    def __init__(self, config: dict, logger):
        self.logger = logger


        try:
            self.config = AppConfig(**config)
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise

    def setup_output_dirs(self):
        """Creates the folders where the model and metrics will be saved.

        If the folders already exist, nothing happens (safe to call every run).
        """
        output_paths = [
             self.config.paths.model_path,
             self.config.paths.metrics_path            
         ]

        for path_str in output_paths:
             path = Path(path_str)

             path.parent.mkdir(parents=True, exist_ok=True)
             self.logger.info(f"Directory ensured: {path.parent}")

    def save_metrics(self, metrics: dict, path: Path):
        """Saves the evaluation results to a JSON file.

        Args:
            metrics (dict): A dictionary of metric names and their values.
            path (Path): Where to save the JSON file.
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Metrics were recorded successfully.: {path}")
        except Exception as e:
            self.logger.error(f"An error occurred while recording the metrics.: {e}")
            raise

    def run_training(self) -> tuple[Pipeline, dict[str, Any]]:
        """Runs the complete training process from start to finish.

        What this method does:
            1. Creates output folders if they don't exist.
            2. Loads the cleaned CSV dataset.
            3. Splits data into 80% training and 20% test sets.
            4. Builds the preprocessing pipeline (scaling + encoding + features).
            5. Loads the classifier defined in config.yaml.
            6. Trains the full pipeline on the training set.
            7. Makes predictions on the test set.
            8. Calculates evaluation metrics (F1, AUC, etc.).
            9. Saves metrics to a JSON file.
            10. Saves the trained pipeline to disk with joblib.

        Returns:
            tuple: The trained pipeline and a dictionary of evaluation metrics.
        """
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


        # stratify=y ensures both train and test have the same churn ratio
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


        for k, v in params.items():
            if v == "None":
                params[k] = None

        
        # Load the model class from its module path (e.g. sklearn.ensemble.RandomForestClassifier)
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


        cv_module_name = self.config.hyperparameters.module
        cv_model_name = self.config.hyperparameters.model_name
        raw_param_grid = self.config.hyperparameters.params

        for k, v_list in raw_param_grid.items():
            if isinstance(v_list, list):
                raw_param_grid[k] = [None if val == "None" else val for val in v_list]

        param_grid = {f"model__{k}": v for k, v in raw_param_grid.items()}


        try:
            cv_module = importlib.import_module(cv_module_name)
            cv_class = getattr(cv_module, cv_model_name)

            self.logger.info(f"Initializing {cv_model_name}...")

            search_model = cv_class(
                estimator=full_model_pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=1,
                verbose=1  # To speed up the training process, you can set the value to -1 to use all CPU cores
            )

            self.logger.info("Pipeline training with {cv_model_name} (Fit process)...")
            self.logger.info(f"Total parameter combinations to try: {sum([len(v) for v in param_grid.values()])}")


            search_model.fit(X_train, y_train)

            self.logger.info(f"Best parameters found: {search_model.best_params_}")

            full_model_pipeline = search_model.best_estimator_
        except ValueError as e:
            self.logger.error(f"Data error during model training (X or y may be mismatched): {e}")
            raise
        except Exception as e:
            self.logger.error(f"A technical problem occurred while training the model: {e}")
            raise



        y_pred = full_model_pipeline.predict(X_test)



        # Some metrics (like ROC-AUC) need probability scores, not just 0/1 predictions
        if hasattr(full_model_pipeline, "predict_proba"):
            y_prob = full_model_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        

        raw_feature_names = full_model_pipeline.named_steps["prep_steps"].named_steps["data_preprocessing"].get_feature_names_out()
        feature_names = [name.split("__")[-1] for name in raw_feature_names]

        trained_model = full_model_pipeline.named_steps["model"]


        visualizer = ModelVisualizer(self.logger, self.config.paths.visualizer_path)
        visualizer.plot_confusion_matrix(y_test, y_pred)
        visualizer.plot_feature_importance(trained_model, feature_names, 10)
        visualizer.plot_precision_recall_curve(y_test, y_prob)



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
