from pathlib import Path
from src.logger import setup_logger
from src.utils import load_config
from src.data_loader import DataLoader
from src.train import ModelTrainer


def main():
    
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "config.yaml"

    config = load_config(config_path)

    logger = setup_logger()

    logger.info("ML Pipeline is being launched...")

    data = DataLoader(config=config, logger=logger)
    model = ModelTrainer(config= config, logger=logger)

    data.load_csv()
    model.run_training()



if __name__ == "__main__":
    main()

