from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {
        "message": "Diabetes Readmission Prediction API is running."
    }
