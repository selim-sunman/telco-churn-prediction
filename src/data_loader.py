"""
data_loader.py - Loads and cleans the raw Telco dataset.

This module reads the raw CSV file, removes unnecessary columns,
fixes data types, handles missing values, and saves the clean
version so the rest of the pipeline can use it.
"""

import pandas as pd
from pathlib import Path
from src.schemas import AppConfig


class DataLoader:
    """Reads the raw CSV file, cleans it, and saves the result.

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

    def load_csv(self) -> pd.DataFrame:
        """Loads and cleans the raw dataset, then returns it as a DataFrame.

        Steps:
            1. Reads the CSV from the path in config.yaml.
            2. Drops the customerID column (not useful for predictions).
            3. Removes extra whitespace from column names and text values.
            4. Converts TotalCharges to a number (some rows have blank values).
            5. Removes rows with missing values and logs which columns were affected.
            6. Removes duplicate rows.
            7. Saves the clean data to the processed folder.
        """


        
        self.logger.info(f"Loading raw data from: {self.config.paths.raw_path}")


        try:
            raw_df = pd.read_csv(self.config.paths.raw_path)
        except FileNotFoundError as e:
            self.logger.error(f"Raw data file not found at: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read data from {self.config.paths.raw_path}: {e}")
            raise


        raw_df = raw_df.drop("customerID", axis=1, errors="ignore")

        
        raw_df.columns = raw_df.columns.str.strip()
        
        obj_cols = raw_df.select_dtypes(include="object").columns
        raw_df[obj_cols] = raw_df[obj_cols].apply(lambda x: x.str.strip())
        self.logger.info("Whitespace cleaned")


        # Some TotalCharges values are blank strings — convert them to NaN so pandas handles them correctly
        raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
        


        missing_value = raw_df.isnull().sum()
        missing_cols = missing_value[missing_value > 0]

        if not missing_cols.empty: 
            self.logger.warning(
                f"Missing values detected: {missing_cols.to_dict()}"
                )
            raw_df = raw_df.dropna(subset=["TotalCharges"])
            self.logger.info(f"The {len(missing_value)} row containing missing data has been deleted.")
        else:
            self.logger.info("No missing values found. Dataset is clean")


        raw_df = raw_df.drop_duplicates()
        self.logger.warning("Duplicate lines have been removed.")



        try:
            output_path = Path(self.config.paths.processed_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned data saved to {self.config.paths.processed_path} : {raw_df.duplicated().sum()}")
        except OSError as e:
            self.logger.error(f"Failed to save cleaned data to {self.config.paths.processed_path}. Check disk space or permissions: {e}")
            raise



        return raw_df

