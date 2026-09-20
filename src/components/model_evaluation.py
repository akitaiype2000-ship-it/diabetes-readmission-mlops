import os
import json
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):

        logger.info("Starting Model Evaluation")

        # Load trained model
        model = joblib.load(self.config.model_path)

        # Load transformed test data
        test = np.load(
            self.config.test_data_path,
            allow_pickle=True
        )

        # Split features and target
        X_test = test[:, :-1]
        y_test = test[:, -1]

        # Predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        # Store metrics
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        # Create output directory
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save metrics.json
        with open(
            self.config.metrics_file_name,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # Classification Report
        report = classification_report(
            y_test,
            predictions,
            zero_division=0
        )

        with open(
            self.config.classification_report_file,
            "w"
        ) as file:

            file.write(report)

        logger.info("Model Evaluation Completed")

        print("\n==============================")
        print("MODEL EVALUATION")
        print("==============================")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("==============================")
        print("Metrics saved successfully.")