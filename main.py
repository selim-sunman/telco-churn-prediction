from pathlib import Path
from src.logger import setup_logger
from src.utils import load_config
from src.data_loader import DataLoader
from src.train import ModelTrain





def main():
    
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "config.yaml"

    logger = setup_logger()

    logger.info("ML Pipeline is being launched...")

    config = load_config(config_path)

    data = DataLoader(config, logger)

    data.load_csv()


    train = ModelTrain(config=config, logger=logger)

    train.run_training()

if __name__ == "__main__":
    main()

