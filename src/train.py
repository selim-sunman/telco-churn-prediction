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
import json
import joblib
import importlib
import pandas as pd
from typing import Any
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import PreprocessingPipeline
from src.evaluation import ModelEvaluator
from src.schemas import AppConfig
from src.visualizer import ModelVisualizer



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

    def _setup_output_dirs(self):
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

    def _load_dataset(self) -> pd.DataFrame:
        """Loads the processed CSV dataset and returns it as a DataFrame."""
        data_path = Path(self.config.paths.processed_path)
        self.logger.info("Data is being loaded and separated into Train/Test sections...")

        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError(f"The dataset read is empty: {data_path}")
            
            return df
        
        except FileNotFoundError:
            self.logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading data: {e}")
            raise

    def _split_data(self, df: pd.DataFrame):
        """Splits the dataset into stratified train/test sets."""
        target = self.config.train_settings.target_col
        X = df.drop(columns=[target])
        y = df[target].map({"Yes": 1, "No": 0})

        self.logger.info("Splitting data into train/test sets (stratified)...")
        # stratify=y ensures both train and test have the same churn ratio
        return train_test_split(
            X, y,
            test_size=self.config.train_settings.test_size,
            random_state=self.config.train_settings.random_state,
            stratify=y
        )
        
    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_preproceessing(self) -> Pipeline:
        """Builds the preprocessing portion of the pipeline."""
        self.logger.info("Building preprocessing steps...")
        preprocessing_pipeline = PreprocessingPipeline()


        return preprocessing_pipeline.create_pipeline(
            numerical_cols=self.config.preprocessing.numerical_cols,
            categorical_cols=self.config.preprocessing.categorical_cols,
            service_cols=self.config.preprocessing.service_cols
        )

    def _load_model(self):
        """Dynamically loads the model class defined in config.yaml."""
        module_name = self.config.model.module
        model_name = self.config.model.model_name
        params = self.config.model.params


        for k, v in params.items():
            if v == "None":
                params[k] = None

        self.logger.info(f"Loading model: {module_name}.{model_name}")
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, model_name) 
            return model_class(**params)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Model library or class not found: {e}")
            raise
    
    def _build_full_pipeline(self) -> Pipeline:
        """Combines preprocessing and the classifier into one Pipeline."""
        preprocessing_steps = self._build_preproceessing()
        model = self._load_model()

        return Pipeline(steps=[
            ("prep_steps", preprocessing_steps),
            ("model", model)
        ])
    
    # ------------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------------

    def _hyperparameter_search(self, full_pipeline: Pipeline, X_train, y_train) -> Pipeline:
        """Runs hyperparameter search (e.g. GridSearchCV) and returns the best estimator."""
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
                estimator=full_pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=1,
                verbose=1  # To speed up the training process, you can set the value to -1 to use all CPU cores
            )

            self.logger.info(f"Pipeline training with {cv_model_name} (Fit process)...")
            self.logger.info(f"Total parameter combinations to try: {sum([len(v) for v in param_grid.values()])}")


            search_model.fit(X_train, y_train)

            self.logger.info(f"Best parameters found: {search_model.best_params_}")
            return search_model.best_estimator_
        
        except ValueError as e:
            self.logger.error(f"Data error during model training (X or y may be mismatched): {e}")
            raise
        except Exception as e:
            self.logger.error(f"A technical problem occurred while training the model: {e}")
            raise

    def _predict(self, pipeline: Pipeline, X_test):
        """Generates label predictions and (when available) probability scores."""
        y_pred = pipeline.predict(X_test)

        # Some metrics (like ROC-AUC) need probability scores, not just 0/1 predictions
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        return y_pred, y_prob

    def _generate_visualizations(self, pipeline: Pipeline, y_test, y_pred, y_prob) -> None:
        """Draws all evaluation charts to the configured figures folder."""
        raw_feature_names = (
            pipeline.named_steps["prep_steps"]
                .named_steps["data_preprocessing"]
                .get_feature_names_out()
        )
        feature_names = [name.split("__")[-1] for name in raw_feature_names]
        trained_model = pipeline.named_steps["model"]

        visualizer = ModelVisualizer(self.logger, self.config.paths.visualizer_path)
        visualizer.plot_confusion_matrix(y_test, y_pred)
        visualizer.plot_feature_importance(trained_model, feature_names, 10)
        if y_prob is not None:
            visualizer.plot_precision_recall_curve(y_test, y_prob)

    def _evaluate(self, y_test, y_pred, y_prob) -> dict[str, Any]:
        """Runs the metrics defined in config.yaml against the predictions."""
        self.logger.info("The Evaluation module is being called for model evaluation...")
        evaluator = ModelEvaluator(metrics_config=self.config.metrics, logger=self.logger)

        return evaluator.evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)

    def _save_metrics(self, metrics: dict, path: Path) -> None:
        """Saves the evaluation results to a JSON file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Metrics were recorded successfully.: {path}")
        except Exception as e:
            self.logger.error(f"An error occurred while recording the metrics.: {e}")
            raise

    def _save_model(self, pipeline: Pipeline, path: Path) -> None:
        """Persists the trained pipeline to disk using joblib."""
        try:
            self.logger.info(f"Saving the model: {path}")
            joblib.dump(pipeline, path, compress=3)
        except OSError as e:
            self.logger.error(f"Disk error! Model could not be saved: {e}")
            raise

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_training(self) -> tuple[Pipeline, dict[str, Any]]:
        """Runs the complete training process from start to finish.
 
        High-level steps:
            1. Ensure output folders exist.
            2. Load the cleaned dataset and split into train/test.
            3. Build a pipeline (preprocessing + classifier).
            4. Hyperparameter-search to find the best estimator.
            5. Predict on the test set + draw evaluation charts.
            6. Compute metrics, then persist metrics and model to disk.
 
        Returns:
            (trained_pipeline, metrics_dict)
        """
        self._setup_output_dirs()

        df = self._load_dataset()
        X_train, X_test, y_train, y_test = self._split_data(df)

        full_pipeline = self._build_full_pipeline()
        best_pipeline = self._hyperparameter_search(full_pipeline,X_train, y_train)

        y_pred, y_prob = self._predict(best_pipeline, X_test)
        self._generate_visualizations(best_pipeline, y_test, y_pred, y_prob)

        metrics_results = self._evaluate(y_test, y_pred, y_prob)

        self._save_metrics(metrics_results, Path(self.config.paths.metrics_path))
        self._save_model(best_pipeline, Path(self.config.paths.model_path))

        return best_pipeline, metrics_results
