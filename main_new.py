from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import serving
import numpy as np
import pandas as pd

app = FastAPI(title="Telco Churn API")

class Customer(BaseModel):
    # Minimal required fields only
    gender: Optional[str] = "Male"
    SeniorCitizen: Optional[int] = 0
    Partner: Optional[str] = "No"
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def predict(customer: Customer) -> dict:
    try:
        # Convert to dict + fill defaults for model
        data = customer.dict()
        # Minimal preprocessing to match trained model
        data['TotalCharges'] = float(data["TotalCharges"])

        result = serving.predictor.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# def predict(request: dict):
#     try:
#         # Handle any JSON input
#         data = request.copy()
#         data['tenure'] = int(data.get('tenure', 1))
#         data['MonthlyCharges'] = float(data.get('MonthlyCharges', 50.0))
#         data['TotalCharges'] = float(data.get('TotalCharges', 500.0))

#         result = serving.predictor.predict(data)
#         return result
#     except Exception as e:
#         return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)