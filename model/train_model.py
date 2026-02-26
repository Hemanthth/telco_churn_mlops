import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

mlflow.set_experiment("telco_churn_experiment")
mlflow.autolog()

def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler

with mlflow.start_run():
    #Load
    df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
    df, scaler = preprocess(df)

    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    mlflow.log_metric("roc_auc", auc)
    print(classification_report(y_test, y_pred))

    # Save to registry
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    mlflow.sklearn.load_model(model, "model")
    print("Model saved to model/model.pkl")


# After training transition to higher stages
client = mlflow.MlflowClient()
model_name = "telco_churn_model"

# Transition stages
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Staging"
)

# Manual approval -> Production
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Production"
)
