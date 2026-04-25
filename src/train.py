import pandas as pd
from sklearn.model_selection import train_test_split



class model_train:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger


    def run_training(self):
        


        try:
            interim_path = self.config["data"]["interim_path"]
            processed_path = self.config["data"]["processed_path"]
            target_col = self.config["train_settings"]["target_col"]
        except KeyError as e:
            self.logger.error(f"Configuration key missing: {e}")
            raise


        df = pd.read_csv(interim_path)


        X = df.drop(columns=[target_col])
        y = df[target_col].map({"Yes": 1, "No": 0})

        X_train, X_test, y_train, y_est = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

