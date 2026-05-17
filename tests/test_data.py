import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import DataLoader


def test_data_loader(sample_raw_data, dummy_config, mock_logger):
    """Basic happy path: clean data is non-empty, customerID is gone, TotalCharges is numeric."""
    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)

    data_load = DataLoader(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()

    assert len(clean_data) > 0
    assert "customerID" not in clean_data.columns
    assert clean_data["TotalCharges"].dtype == "float64"


def test_missing_data_clean(sample_raw_data, dummy_config, mock_logger):
    """Rows with a blank TotalCharges value should be dropped."""
    sample_raw_data.loc[0, "TotalCharges"] = " "

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)

    data_load = DataLoader(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()

    assert len(clean_data) == len(sample_raw_data) - 1
    assert clean_data.isnull().sum().sum() == 0
    
def test_duplicates_are_removed(sample_raw_data, dummy_config, mock_logger):
    """Duplicate rows in the source CSV must be deduplicated by load_csv()."""
    duplicated_df = sample_raw_data.copy()

    duplicated_df.iloc[1, 1:] = duplicated_df.iloc[0, 1:].values

    file_path = dummy_config["paths"]["raw_path"]
    duplicated_df.to_csv(file_path, index=False)

    data_load = DataLoader(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()
 
    assert len(clean_data) < len(duplicated_df)
    assert clean_data.duplicated().sum() == 0


def test_file_error(sample_raw_data, dummy_config, mock_logger):
    """Missing raw file should raise FileNotFoundError."""
    file_ = Path(dummy_config["paths"]["raw_path"])
    if file_.exists():
        file_.unlink()

    data_load = DataLoader(config=dummy_config, logger=mock_logger)


    with pytest.raises(FileNotFoundError):
        data_load.load_csv()


def test_processed_file_is_written_correctly(sample_raw_data, dummy_config, mock_logger):
    """The processed CSV on disk must match the in-memory cleaned DataFrame."""
    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)
 
    processed_path = Path(dummy_config["paths"]["processed_path"])
    if processed_path.exists():
        processed_path.unlink()
 
    data_load = DataLoader(config=dummy_config, logger=mock_logger)
    clean_data = data_load.load_csv()
 
    assert processed_path.exists()
 
    saved_data = pd.read_csv(processed_path)
    assert len(saved_data) == len(clean_data)
    assert list(saved_data.columns) == list(clean_data.columns)
    assert "customerID" not in saved_data.columns
    assert "customerID" not in clean_data.columns




    