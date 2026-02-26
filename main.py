from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
import serving

app = FastAPI(title="Telco Churn API")

# class Customer(BaseModel):
#     customerID: str
#     gender: str
#     SeniorCitizen: int
#     Partner: str
#     Dependents: str
#     tenure: int
#     PhoneService: str
#     MultipleLines: str
#     InternetService: str
#     OnlineSecurity: str
#     OnlineBackup: str
#     DeviceProtection: str
#     TechSupport: str
#     StreamingTV: str
#     StreamingMovies: str
#     Contract: str
#     PaperlessBilling: str
#     PaymentMethod: str
#     MonthlyCharges: float
#     TotalCharges: float
#     Churn: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    try:
        predictor = serving.create_predictor()
        result = predictor.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)