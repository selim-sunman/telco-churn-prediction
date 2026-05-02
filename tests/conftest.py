import pytest
import pandas as pd
import yaml



class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def exception(self, msg): pass



@pytest.fixture
def mock_logger():
    return DummyLogger()


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "customerID": [str(i) for i in range(1, 11)],
        "gender": ["Female", "Male", "Male", "Female", "Male", "Female", "Female", "Male", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "Partner": ["Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45, 2, 8, 22, 10, 28, 62],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service", "No", "Yes", "Yes", "No phone service", "Yes", "No"],
        "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic", "Fiber optic", "DSL", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No", "No", "Yes", "No", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "Yes"],
        "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes", "Yes"],
        "TechSupport": ["No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],
        "StreamingTV": ["No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year", "Month-to-month", "Month-to-month", "Two year", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Electronic check", "Credit card (automatic)", "Mailed check", "Electronic check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 56.15],
        "TotalCharges": ["29.85", "1889.5", "108.15", "1840.75", "151.65", "820.5", "1949.4", "301.9", "3046.05", "3487.95"],
        "Churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"]
    })


@pytest.fixture
def dummy_config(tmp_path):
    raw_data_path = tmp_path / "raw_data.csv"
    interim_data_path = tmp_path / "interim_data.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"



    ayarlar = {
        "paths": {
            "raw_path": str(raw_data_path),
            "interim_path": str(interim_data_path),
            "train_path": str(train_path),
            "test_path": str(test_path)
        },
        "train_settings": {
            "target_col": "Churn",
            "test_size": 0.2,
            "random_state": 42
        },
        "preprocessing": {
            "numerical_cols": ["tenure", "MonthlyCharges", "TotalCharges", "TotalService", "TotalCharges_log"],
            "categorical_cols": ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "PaperlessBilling", "PaymentMethod", "HasFamily"],
            "service_cols": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        },
        "model": {
            "module": "sklearn.linear_model",
            "model_name": "LogisticRegression",
            "params": {
                "max_iter": 100
            }
        },
        "metrics": [
            {"module": "sklearn.metrics", "name": "accuracy_score"},
            {"module": "sklearn.metrics", "name": "roc_auc_score"}
        ]
    }
    
    return ayarlar



@pytest.fixture
def yaml_fie_path(tmp_path, config_path):

    file_path = tmp_path / "config.yaml"

    with open(file_path, "w", encoding='utf-8') as f:
        yaml.dump(config_path, f)

    return str(file_path)