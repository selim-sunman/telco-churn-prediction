from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from src.utils import load_config
from src.logger import setup_logger
from pathlib import Path
import joblib
import pandas as pd


logger = setup_logger()

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "config.yaml"


    try:
        config = load_config(config_path)
        model_path = config["paths"]["model_path"]

        app_state["pipeline"] = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
    except Exception as e:
        logger.error(f"A critical error occurred while loading the model: {e}")
        raise

    yield

    app_state.clear()
    logger.info("API Closed: Model pipeline in memory cleared.")



app = FastAPI(
    title = "Telco Churn Prediction API",
    description = "An ML API that predicts the likelihood of customers churning.",
    version="1.0.0",
    lifespan=lifespan
)



class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer stayed with the company")
    SeniorCitizen: int = Field(..., ge=0, description="Is the customer elderly? (1: Yes, 0: No)")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly salary")  
    TotalCharges: float = Field(..., ge=0.0, description="Total amount paid")


    gender: str = Field(..., description="gender (Male, Female)")
    Partner: str = Field(..., description="Does he/she have a partner? (Yes, No)")
    Dependents: str = Field(..., description="Does he/she have someone he/she is responsible for supporting? (Yes, No)")
    PhoneService: str = Field(..., description="Phone service (Yes, No)")
    MultipleLines: str = Field(..., description="Multiple lines (Yes, No, No phone service))")    
    InternetService: str = Field(..., description="Internet service (DSL, Fiber optic, No))")
    OnlineSecurity: str = Field(..., description="Online security (Yes, No, No internet service)")    
    OnlineBackup: str = Field(..., description="Online backup (Yes, No, No internet service)")      
    DeviceProtection: str = Field(..., description="Device protection (Yes, No, No internet service)")   
    TechSupport: str = Field(..., description="Technical support (Yes, No, No internet service)")        
    StreamingTV: str = Field(..., description="TV Broadcast (Yes, No, No internet service)")       
    StreamingMovies: str = Field(..., description="Movie Streaming (Yes, No, No internet service)")   
    Contract: str = Field(..., description="Contract type (Month-to-month, One year, Two year)")          
    PaperlessBilling: str = Field(..., description="Paperless invoice (Yes, No)") 
    PaymentMethod: str = Field(..., description="Payment method")


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


@app.get("/", tags=["Health"])
def root():
    return {"status": "healthy", "message": "Telco Churn Prediction API is active and working!"}


@app.post("/predict", tags=["prediction"])
def predict_churn(customer: CustomerData):

    pipeline = app_state.get("pipeline")

    if pipeline is None:
        raise HTTPException(status_code=500, detail="The model has not been uploaded yet or could not be found.")
    
    try:

        input_df = pd.DataFrame([customer.dict()])

        logger.info(f"Data retrieved for estimation: {customer.tenure} months, ${customer.MonthlyCharges} monthly charge.")

        prediction_val = pipeline.predict(input_df)[0]

        probability_val = 0.0
        if hasattr(pipeline, "predict_proba"):
            probability_val = pipeline.predict_proba(input_df)[0][1]

        result_label = "Yes" if prediction_val == 1 else "No"

        logger.info(f"Prediction result: Churn={result_label} (Probability: {probability_val:.2f})")


        return {
            "churn_prediction": result_label,
            "churn_probability": round(probability_val, 2)
        }
    
    except Exception as e:
        logger.error(f"Error during estimation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    
