"""
main.py - Entry point for the Telco Churn Prediction pipeline.
 
Running this file executes the full training workflow end-to-end:
 
    1. Read project settings from ``config/config.yaml``.
    2. Set up the logger so progress messages are printed and saved.
    3. Load and clean the raw Telco dataset (DataLoader).
    4. Build the preprocessing + model pipeline, run hyperparameter
       search, evaluate the model, and persist the artifacts (ModelTrainer).
 
Usage
-----
From the project root::
 
    python main.py
 
After the run finishes you can inspect:
    * ``reports/metrics.json``   - evaluation scores
    * ``reports/figures/``       - confusion matrix, PR curve, feature importance
    * ``models/model.joblib``    - the trained pipeline
    * ``logs/app_*.log``         - the full execution log
"""

from pathlib import Path
from src.logger import setup_logger
from src.utils import load_config
from src.data_loader import DataLoader
from src.train import ModelTrainer


def main():
    """Run the complete churn-prediction training pipeline.
 
    Steps performed in order:
        * Resolve the config path relative to this file (so the script
          works no matter what directory it is launched from).
        * Validate and load ``config.yaml`` as a plain Python dict.
        * Initialise the logger (colored stdout + rotating file log).
        * Load and clean the raw CSV through ``DataLoader``.
        * Train, tune, evaluate and persist the model through ``ModelTrainer``.
    """
    # Resolve config path relative to this file rather than the current
    # working directory — this makes `python main.py` work from anywhere.
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "config.yaml"

    config = load_config(config_path)
    logger = setup_logger()

    logger.info("ML Pipeline is being launched...")

    # DataLoader cleans the raw CSV and writes the processed file
    # that ModelTrainer will read from in the next step.
    data = DataLoader(config=config, logger=logger)
    model = ModelTrainer(config= config, logger=logger)

    data.load_csv()
    model.run_training()



if __name__ == "__main__":
    main()

