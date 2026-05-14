"""
sql_pipeline.py - Demonstrates using SQL within Python for Data Engineering.

This module shows how to load raw CSV data into an in-memory SQLite database,
perform data cleaning and feature engineering using advanced SQL queries,
and return the result as a Pandas DataFrame ready for machine learning.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path

class SQLDataLoader:
    """
    Loads Telco data and transforms it using SQL.
    """
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        # Create an in-memory SQLite database
        self.conn = sqlite3.connect(":memory:") 
        self.logger = logging.getLogger(__name__)

    def load_and_transform(self) -> pd.DataFrame:
        """
        Loads the CSV, pushes it to SQLite, runs a feature engineering SQL query,
        and returns the cleaned pandas DataFrame.
        """
        # 1. Load CSV into pandas temporarily just to dump into SQLite
        try:
            self.logger.info(f"Loading raw data from: {self.raw_data_path}")
            raw_df = pd.read_csv(self.raw_data_path)
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
            
        # 2. Write DataFrame to SQLite table named 'telco_raw'
        raw_df.to_sql('telco_raw', self.conn, index=False, if_exists='replace')
        self.logger.info("Data loaded into SQLite in-memory database.")

        # 3. Define the SQL Query for Cleaning and Feature Engineering
        sql_query = """
        WITH CleanedData AS (
            -- Step 1: Clean missing values and format data types
            SELECT 
                customerID,
                gender,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure,
                PhoneService,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                -- Handle blank TotalCharges by converting to NULL then FLOAT
                CAST(NULLIF(TRIM(TotalCharges), '') AS FLOAT) AS TotalCharges,
                -- Convert target to binary
                CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS Churn_Target
            FROM telco_raw
        ),
        FeatureEngineering AS (
            -- Step 2: Create new features directly in SQL
            SELECT 
                *,
                -- Feature 1: Has Family (If Partner or Dependents is Yes)
                CASE WHEN Partner = 'Yes' OR Dependents = 'Yes' THEN 1 ELSE 0 END AS HasFamily,
                
                -- Feature 2: Total number of extra services
                (
                    (CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END) +
                    (CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END)
                ) AS TotalServices,
                
                -- Feature 3: Tenure Groups (Categorization via SQL)
                CASE 
                    WHEN tenure <= 12 THEN '0-1 Year'
                    WHEN tenure > 12 AND tenure <= 24 THEN '1-2 Years'
                    WHEN tenure > 24 AND tenure <= 48 THEN '2-4 Years'
                    ELSE '4+ Years' 
                END AS TenureGroup
            FROM CleanedData
            -- Step 3: Remove rows where TotalCharges was blank (now NULL)
            WHERE TotalCharges IS NOT NULL
        )
        
        -- Step 4: Final Selection
        SELECT * FROM FeatureEngineering;
        """

        self.logger.info("Executing SQL transformation...")
        
        # 4. Execute the query and load results directly into a new Pandas DataFrame
        processed_df = pd.read_sql_query(sql_query, self.conn)
        
        # Drop customerID at the end since it's not needed for the ML model
        if 'customerID' in processed_df.columns:
            processed_df = processed_df.drop('customerID', axis=1)
            
        self.logger.info(f"SQL Transformation completed. Resulting shape: {processed_df.shape}")
        
        return processed_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Path resolution for standalone running
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "data" / "raw" / "dataset.csv"
    
    loader = SQLDataLoader(raw_data_path=str(data_path))
    final_df = loader.load_and_transform()
    
    print("\n--- İlk 5 Satır (SQL ile İşlenmiş) ---")
    print(final_df[['gender', 'tenure', 'TotalCharges', 'HasFamily', 'TotalServices', 'TenureGroup', 'Churn_Target']].head())
