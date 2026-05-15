from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils import load_config
from src.logger import setup_logger
from pathlib import Path
import joblib
import pandas as pd


app = FastAPI(
    title = "Telco Churn Prediction API",
    description = "An ML API that predicts the likelihood of customers churning.",
    version="1.0.0"
)



class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    Phoneservice: str
    MultipleLines: str     
    InternetService: str  
    OnlineSecurity: str     
    OnlineBackup: str       
    DeviceProtection: str   
    TechSupport: str         
    StreamingTV: str        
    StreamingMovies: str    
    Contract: str           
    PaperlessBilling: str  
    PaymentMethod: str      
    MonthlyCharges: float     
    TotalCharges: float


    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "SeniorCitizen": 0,
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }



model_path = None
model_pipeline = None

logger = setup_logger()

@app.on_event("startup")
def load_model():

    global model_path
    global model_pipeline

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"

    config = load_config(config_path)


    try:
        model_path = config["paths"]["model_path"]

        model_pipeline = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
    except Exception as e:
        logger.error(f"A critical error occurred while loading the model: {e}")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "The Telco Churn API Is Up and Running!"}


@app.post("/predict", tags=["prediction"])
def predict_churn(customer: CustomerData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="The model has not been uploaded yet or could not be found.")
    
    try:

        input_df = pd.DataFrame([customer.model_dump()])

        logger.info(f"Data retrieved for estimation: {customer.tenure} months, ${customer.MonthlyCharges} monthly charge.")

        prediction_val = model_pipeline.predict(input_df)

        probability_val = 0.0
        if hasattr(model_pipeline, "predict_proba"):
            probability_val = model_pipeline.predict_proba(input_df)[0][1]

        result_label = "Yes" if prediction_val == 1 else "No"

        logger.info(f"Prediction result: Churn={result_label} (Probability: {probability_val:.2f})")


        return {
            "churn_prediction": result_label,
            "churn_probability": round(probability_val, 2)
        }
    
    except Exception as e:
        logger.error(f"Error during estimation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
