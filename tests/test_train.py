import pytest
import pandas as pd
import json
from pathlib import Path
from src.train import ModelTrainer
from src.data_loader import DataLoader




def test_training_works(sample_raw_data, dummy_config, mock_logger):

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)
    
    data_loader_obj = DataLoader(config=dummy_config, logger=mock_logger)
    data_loader_obj.load_csv()

    trainer = ModelTrainer(config=dummy_config, logger=mock_logger)
    trained_pipeline, metric_results = trainer.run_training()
    
 

    from sklearn.pipeline import Pipeline
    assert isinstance(trained_pipeline, Pipeline)
    

    assert isinstance(metric_results, dict)
    assert len(metric_results) > 0 
    assert "accuracy_score" in metric_results 
    

    train_file = Path(dummy_config["paths"]["train_path"])
    test_file = Path(dummy_config["paths"]["test_path"])
    
    assert train_file.exists(), "Train verisi bilgisayara kaydedilemedi!"
    assert test_file.exists(), "Test verisi bilgisayara kaydedilemedi!"
    
  
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    target_col_name = dummy_config["train_settings"]["target_col"]
    assert target_col_name in train_df.columns
    assert target_col_name in test_df.columns

    model_file = Path(dummy_config["paths"]["model_path"])
    metrics_file = Path(dummy_config["paths"]["metrics_path"])

    assert model_file.exists(), "The model file (.joblib) could not be saved!"
    assert metrics_file.exists(), "Could not save metric file (.json)!"

    with open(metrics_file, "r") as f:
        saved_metrics = json.load(f)
        assert len(saved_metrics) > 0
    



def test_raises_error_with_empty_dataframe(dummy_config, mock_logger):

    target_col = dummy_config["train_settings"]["target_col"]
    empty_df = pd.DataFrame(columns=[target_col, "some_other_col"])

    file_path = dummy_config["paths"]["interim_path"]
    empty_df.to_csv(file_path, index=False)

    trainer = ModelTrainer(config=dummy_config, logger=mock_logger)

    with pytest.raises(ValueError, match="The dataset read is empty"):
        trainer.run_training()






def test_raises_error_with_wrong_module_name(sample_raw_data, dummy_config, mock_logger):
 
    

    file_path = dummy_config["paths"]["raw_path"]
    sample_raw_data.to_csv(file_path, index=False)
    data_loader_obj = DataLoader(config=dummy_config, logger=mock_logger)
    data_loader_obj.load_csv()
    

    broken_config = dummy_config.copy()
    broken_config["model"]["module"] = "sklearn.no_such_module_exists_i_made_it_up"
    
    trainer = ModelTrainer(config=broken_config, logger=mock_logger)
    

    with pytest.raises(ImportError):
        trainer.run_training()
