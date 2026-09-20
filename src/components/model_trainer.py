import os
import json
import joblib
import numpy as np

import mlflow
import mlflow.sklearn

from src.mlflow_utils import setup_mlflow
from src.logger import logger
from src.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        # Load transformed datasets
        train = np.load(
            self.config.train_data_path,
            allow_pickle=True
        )

        test = np.load(
            self.config.test_data_path,
            allow_pickle=True
        )

        logger.info("Transformed data loaded")

        # Split features and target
        X_train = train[:, :-1]
        y_train = train[:, -1]

        X_test = test[:, :-1]
        y_test = test[:, -1]

        # Models to compare
        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=3000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

            "XGBoost":
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ),

            "LightGBM":
                LGBMClassifier(
                    random_state=42
                )
        }

        setup_mlflow()

        results = {}

        best_model = None
        best_model_name = ""
        best_f1 = 0
                # Train every model
        for name, model in models.items():

            print(f"\nTraining {name}...")

            with mlflow.start_run(run_name=name):

                # Train model
                model.fit(
                    X_train,
                    y_train
                )

                # Predictions
                predictions = model.predict(
                    X_test
                )

                # Evaluation metrics
                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                # Log metrics to MLflow
                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "f1_score",
                    f1
                )

                # Log hyperparameters
                mlflow.log_params(
                    model.get_params()
                )

                # Store metrics for comparison
                results[name] = {

                    "accuracy": round(
                        accuracy,
                        4
                    ),

                    "precision": round(
                        precision,
                        4
                    ),

                    "recall": round(
                        recall,
                        4
                    ),

                    "f1_score": round(
                        f1,
                        4
                    )
                }

                print(f"\n{name}")
                print(f"Accuracy : {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall   : {recall:.4f}")
                print(f"F1 Score : {f1:.4f}")

                # Save best model
                if f1 > best_f1:

                    best_f1 = f1
                    best_model = model
                    best_model_name = name

                # Create output folder
        os.makedirs(
            self.config.root_dir,
            exist_ok=True
        )

        # Save best model locally
        joblib.dump(
            best_model,
            self.config.model_path
        )

        logger.info(
            f"Best model saved: {best_model_name}"
        )

        # Log only the best model to MLflow
        with mlflow.start_run(run_name="Best Model"):

            mlflow.log_param(
                "best_model",
                best_model_name
            )

            mlflow.log_metric(
                "best_f1_score",
                best_f1
            )

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="best_model",
                serialization_format="pickle"
            )

        # Save comparison report
        metrics_path = os.path.join(
            self.config.root_dir,
            "model_scores.json"
        )

        with open(metrics_path, "w") as file:

            json.dump(
                results,
                file,
                indent=4
            )

        logger.info(
            "Model comparison report saved successfully."
        )

        print("\n===================================")
        print(f"Best Model : {best_model_name}")
        print(f"Best F1 Score : {best_f1:.4f}")
        print("===================================")

        print("\nModel comparison report saved successfully.")       