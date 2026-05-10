"""
preprocess.py - Feature engineering and preprocessing pipeline.

This module creates new features from the raw data and prepares
the dataset for model training. It scales numbers and encodes
categories so that scikit-learn models can work with them.

There are two classes here:
- FeatureEngineering: adds three new columns to the dataset.
- PreprocessingPipeline: builds the full sklearn pipeline.
"""

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Creates new features from existing columns in the dataset.

    This is a custom sklearn transformer — it works inside a Pipeline
    just like StandardScaler or OneHotEncoder. It adds three new
    columns that help the model make better predictions.

    Attributes:
        service_cols (list[str]): Columns that describe which services
            the customer uses (e.g. OnlineSecurity, StreamingTV).
    """

    def __init__(self, service_cols):
        self.service_cols = service_cols

    def fit(self, X, y=None):
        """Required by sklearn — this transformer doesn't learn anything.

        Args:
            X: Input data (not used).
            y: Target column (not used).

        Returns:
            FeatureEngineering: Returns itself so sklearn can chain steps.
        """
        return self
    
    def transform(self, X):
        """Adds three new columns to the dataset.

        New columns created:
            - HasFamily: 1 if the customer has a partner or dependents, else 0.
              Customers with families tend to churn less.
            - TotalService: How many add-on services the customer uses (0-8).
              More services usually means a more engaged customer.
            - TotalCharges_log: A log-scaled version of TotalCharges.
              This reduces the effect of very large billing values on the model.

        Args:
            X (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: A copy of X with the three new columns added.
        """
        

        logger.info("The feature engineering (transform) process has begun.")

        X_copy = X.copy()
        logger.info(f"Data size before processing:{X_copy.shape}")

            
        try:
            if "Partner" in X_copy.columns and "Dependents" in X_copy.columns:
                X_copy["HasFamily"] = ((X_copy["Partner"] == "Yes") | (X_copy["Dependents"] == "Yes")).astype(int)
                logger.debug("New feature created: 'HasFamily'")
            else:
                logger.warning("Attention: The 'Partner' or 'Dependents' columns are missing! 'HasFamily' could not be created.")



            available_cols = [col for col in self.service_cols if col in X_copy.columns]
                
            if available_cols:
                # "No internet service" and "No phone service" both mean "No" — simplify them
                X_copy[available_cols] = X_copy[available_cols].replace({
                    "No internet service" : "No",
                    "No phone service" : "No"
                })
                logger.debug(f"The service column {len(available_cols)} has been cleaned.")

                X_copy["TotalService"] = X_copy[available_cols].eq("Yes").sum(axis=1)
                logger.debug("A new feature has been created: 'TotalService'")
            else:
                logger.warning("Attention: Service columns not found! 'TotalService' was not calculated.")


                
            if "TotalCharges" in X_copy.columns:
                X_copy["TotalCharges_log"] = np.log1p(X_copy["TotalCharges"])
                logger.debug("A new feature has been created: 'TotalCharges_log'")
            else:
                logger.warning("Attention: Column 'TotalCharges' not found! Log conversion skipped.")


            logger.info(f"Feature engineering completed. Post-processing data size: {X_copy.shape}")

        except Exception as e:
            logger.error(f"A critical error during the feature engineering phase: {str(e)}")
            raise

            

        return X_copy


class PreprocessingPipeline:
    """Builds the full preprocessing pipeline used before model training.

    Combines FeatureEngineering, StandardScaler, and OneHotEncoder
    into a single sklearn Pipeline object. This way, all the same
    transformations applied during training are automatically applied
    when making predictions.
    """

    def __init__(self):
        pass

    def create_pipeline(self, numerical_cols, categorical_cols, service_cols):
        """Builds and returns the preprocessing pipeline.

        The pipeline has two main steps:
            1. feature_engineering — adds HasFamily, TotalService, TotalCharges_log.
            2. data_preprocessing — scales numbers and encodes categories:
               - StandardScaler: scales numerical columns to a similar range.
               - OneHotEncoder: converts text categories into 0/1 columns.

        Args:
            numerical_cols (list[str]): Columns with numeric values to scale.
            categorical_cols (list[str]): Columns with text values to encode.
            service_cols (list[str]): Service columns used to create TotalService.

        Returns:
            sklearn.pipeline.Pipeline: The complete pipeline, ready to be fitted.
        """
        
        logger.info("Preprocessing pipeline setup is starting...")

        numeric_transform = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transform = Pipeline(steps=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output= False))
        ])


        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform, numerical_cols),
                ("cat", categorical_transform, categorical_cols)
            ], remainder= "drop"
        )


        full_pipeline = Pipeline(steps=[
            ("feature_engineering", FeatureEngineering(service_cols=service_cols)),
            ("data_preprocessing", preprocessor)
        ])

        logger.info("The preprocessing pipeline has been successfully created.")
        
        
        return full_pipeline