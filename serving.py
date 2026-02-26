import joblib
# import pandas as pd
import numpy as np
# from typing import List, Dict, Any

class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load("model/model.pkl")
        # self.scaler = joblib.load("model/scaler.pkl")

    # def preprocess(self, data):
        # df = pd.DataFrame([data])
        # df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # df[['tenure', 'MonthlyCharges', 'TotalCharges']] = self.scaler.transform(df[[['tenure', 'MonthlyCharges', 'TotalCharges']]])
        # return df.values

    
    def predict(self, data):
        # X = self.preprocess(data)
        # prob = self.model.predict_proba(X)[11]
        # churn = 1 if prob > 0.5 else 0
        # return {"churn_probability": float(prob), "churn_flag": bool(churn)}

        # Fixed 19-feature vector for Telco churn
        features = np.array([
            0,0,0,0,
            data.get('tenure', 1),
            0,0,0,0,0,0,0,0,0,0,
            data.get('MonthlyCharges', 50),
            data.get('TotalCharges', 500),
            0,0
        ])

        prob = self.model.predict_proba([[features]])[0, 1]
        return {
            "churn_probability": float(prob),
            "churn_flag": prob > 0.5
        }

def create_predictor():
    return ChurnPredictor()