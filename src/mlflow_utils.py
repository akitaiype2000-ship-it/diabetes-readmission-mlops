import mlflow


def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes Readmission Prediction")