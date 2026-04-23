import yaml
from pathlib import Path
from src.logger import setup_logger
from src.data_loader import DataLoad



def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config




def main():
    
    config_path = Path("config/config.yaml")

    logger = setup_logger()

    logger.info("ML Pipeline is being launched...")

    config = load_config(config_path)

    data = DataLoad(config, logger)

    data.load_csv()



if __name__ == "__main__":
    main()

