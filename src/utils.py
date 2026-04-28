import os
import yaml
import joblib
from datetime import datetime
import pandas as pd
from typing import Any, Dict
from pathlib import Path




def load_config(self, config_path: str) -> Dict[str, Any]:
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"An error occurred while reading the YAML file: {e}")
    



def save_model(model: Any, filepath: str, logger:None) -> None:

    try:

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, filepath)
        if logger:
            logger.info(f"Model successfully saved: {filepath}")
    except Exception as e:
        if logger:
            logger.error(f"An error occurred while saving the model: {e}")
        raise




def save_metrics_to_csv(metrics: Dict[str, float], filepath: str, model_name: str = "Unknown", logger=None) -> None:

    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name
        }

        row_data.update(metrics)

        df = pd.DataFrame([row_data])

        file_exists = os.path.isfile(filepath)

        df.to_csv(filepath, mode="a", header=not file_exists, index=False, encoding='utf-8')

        if logger:
            logger.info("Experiment results were successfully added to the CSV: {filepath}")
    except Exception as e:
        if logger:
            logger.error(f"An error occurred while attaching metrics to the CSV file: {e}")
        raise