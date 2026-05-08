import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, service_cols):
        self.service_cols = service_cols


    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        

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
    def __init__(self):
        self



    def create_pipeline(self, numerical_cols, categorical_cols, service_cols):
        
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