import pandas as pd
from src.preprocess import Feature_Engineering, Preprocessing_Pipeline





def test_feature_engineering(sample_raw_data, dummy_config, mock_logger):
    
    df = sample_raw_data.copy()

    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    service_cols = dummy_config["preprocessing"]["service_cols"]

    transformer = Feature_Engineering(logger=mock_logger, service_cols=service_cols)
    new_df = transformer.transform(df)

    assert "HasFamily" in new_df.columns

    first_customer_family_status = new_df.iloc[0]["HasFamily"]
    assert first_customer_family_status == 1

    assert "TotalCharges_log" in new_df.columns


def test_pipeline(sample_raw_data, dummy_config, mock_logger):

    pipeline_creator = Preprocessing_Pipeline(logger=mock_logger)

    numerical_cols = dummy_config["preprocessing"]["numerical_cols"]
    categorical_cols = dummy_config["preprocessing"]["categorical_cols"]
    service_cols = dummy_config["preprocessing"]["service_cols"]


    pipeline = pipeline_creator.create_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        service_cols=service_cols
    )

    from sklearn.pipeline import Pipeline
    assert isinstance(pipeline, Pipeline)

    steps = dict(pipeline.steps)

    assert "feature_engineering" in steps
    assert "data_preprocessing" in steps

