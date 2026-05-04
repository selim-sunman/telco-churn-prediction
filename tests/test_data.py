import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoad


def test_data_loader(sample_raw_data, dummy_config, mock_logger):

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)


    data_load = DataLoad(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()


    assert len(clean_data) > 0
    assert "customerID" not in clean_data.columns
    assert clean_data["TotalCharges"].dtype == "float64"


   


def test_missing_data_clean(sample_raw_data, dummy_config, mock_logger):

    sample_raw_data.loc[0, "TotalCharges"] = " "

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)

    data_load = DataLoad(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()

    assert len(clean_data) == len(sample_raw_data) - 1
    assert clean_data.isnull().sum().sum() == 0
    



def test_file_error(sample_raw_data, dummy_config, mock_logger):

    file_ = Path(dummy_config["paths"]["raw_path"])
    if file_.exists():
        file_.unlink()

    data_load = DataLoad(config=dummy_config, logger=mock_logger)


    with pytest.raises(FileNotFoundError):
        data_load.load_csv()



def test_clean_data_file(sample_raw_data, dummy_config, mock_logger):

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)

    interim_path = Path(dummy_config["paths"]["interim_path"])


    if interim_path.exists():
        interim_path.unlink()


    data_load = DataLoad(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()


    assert interim_path.exists()

    save_data = pd.read_csv(interim_path)
    assert len(clean_data) == len(sample_raw_data)
    assert "customerID" not in save_data.columns




    