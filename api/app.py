from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

import joblib
import numpy as np
print("*********** THIS IS MY APP.PY ***********")
app = FastAPI(title="Diabetes Readmission Prediction API")

Instrumentator().instrument(app).expose(app)

# Load model and preprocessor
model = joblib.load("artifacts/model_trainer/model.pkl")
preprocessor = joblib.load("artifacts/data_transformation/preprocessor.pkl")
print("\n=== PREPROCESSOR ===")
print(preprocessor)

@app.get("/")
def home():
    return {
        "message": "Diabetes Readmission Prediction API is running."
    }

class PredictionInput(BaseModel):
    race: str
    gender: str
    age: str
    weight: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesMed: str


import pandas as pd
import numpy as np

@app.post("/predict")
def predict(data: PredictionInput):
    print(">>> ENTERED PREDICT FUNCTION <<<")
    df = pd.DataFrame([data.model_dump()])

    # Rename fields back to training column names
    df.rename(columns={
        "glyburide_metformin": "glyburide-metformin",
        "glipizide_metformin": "glipizide-metformin",
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
    }, inplace=True)

    df.replace("?", np.nan, inplace=True)
    print("\n========== DATAFRAME ==========")
    print(df)

    print("\n========== DTYPES ==========")
    print(df.dtypes)

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())
    print("\n==========================")
    print(df)
    print("==========================")
    print("\nDataFrame:")
    print(df)
    print("\nColumns:")
    print(df.columns.tolist())
    X = preprocessor.transform(df)

    print("\n========== TRANSFORMED ==========")
    print(X)
    print("Shape:", X.shape)

    if hasattr(X, "toarray"):
        X = X.toarray()

    print("\n========== FIRST 50 FEATURES ==========")
    print(X[0][:50])

    probabilities = model.predict_proba(X)
    prediction = model.predict(X)

    print("\n========== PREDICT PROBA ==========")
    print(probabilities)

    print("\n========== PREDICT ==========")
    print(prediction)

    pred = int(prediction[0])

    if pred == 1:
        result = "Patient is likely to be readmitted within 30 days."
    else:
        result = "Patient is NOT likely to be readmitted within 30 days."

    return {
    "prediction": pred,
    "readmission": result,
    "probability_not_readmitted": round(float(probabilities[0][0]), 4),
    "probability_readmitted": round(float(probabilities[0][1]), 4)
}  