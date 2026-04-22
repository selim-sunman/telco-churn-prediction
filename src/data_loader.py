import pandas as pd
from pathlib import Path



class DataLoad:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger


    def load_csv(self) -> pd.DataFrame:

        try:
            raw_path = self.config["data"]["raw_path"]
            interim_path = self.config["data"]["interim_path"]
            
        except KeyError as e:
            self.logger.error(f"Configuration key missing: {e}")
            raise

        
        self.logger.info(f"Loading raw data from: {raw_path}")


        try:
            df = pd.read_csv(raw_path)
        except FileNotFoundError as e:
            self.logger.error(f"Raw data file not found at: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read data from {raw_path}: {e}")
            raise

        
        df.columns = df.columns.str.strip()
        
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())
        self.logger.info("Whitespace cleaned")


        
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        


        missing_value = df.isnull().sum()
        missing_cols = missing_value[missing_value > 0]

        if not missing_cols.empty: 
            self.logger.warning(
                f"Missing values detected: {missing_cols.to_dict()}"
                )
            df = df.dropna(subset=["TotalCharges"])
            self.logger.info(f"Dropped rows with missing values: {len(missing_cols)} columns affected")
        else:
            self.logger.info("No missing values found. Dataset is clean")


        df = df.drop("customerID", axis=1, errors="ignore")


        try:
            output_path = Path(interim_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Cleaned data saved to {interim_path}")
        except OSError as e:
            self.logger.error(f"Failed to save cleaned data to {interim_path}. Check disk space or permissions: {e}")
            raise



        return df


        
       
        



