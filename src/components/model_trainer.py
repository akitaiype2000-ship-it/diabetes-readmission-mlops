import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.logger import logger
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        train = np.load(self.config.train_data_path, allow_pickle=True)
        test = np.load(self.config.test_data_path, allow_pickle=True)

        logger.info("Transformed data loaded")

        X_train = train[:, :-1]
        y_train = train[:, -1]

        X_test = test[:, :-1]
        y_test = test[:, -1]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy : {accuracy:.4f}")

        joblib.dump(model, self.config.model_path)

        logger.info("Model saved successfully")

        print("Training Completed")