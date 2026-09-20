import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.entity.config_entity import DataTransformationConfig
from src.logger import logger


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self):

        train = pd.read_csv(self.config.train_data_path)
        test = pd.read_csv(self.config.test_data_path)

        logger.info("Dataset Loaded")

        train.replace("?", np.nan, inplace=True)
        test.replace("?", np.nan, inplace=True)

        logger.info("Question marks replaced")

        drop_cols = [
            "encounter_id",
            "patient_nbr"
        ]

        train.drop(columns=drop_cols, inplace=True)
        test.drop(columns=drop_cols, inplace=True)

        logger.info("Identifier columns removed")

        train["readmitted"] = train["readmitted"].apply(
            lambda x: 1 if x == "<30" else 0
        )

        test["readmitted"] = test["readmitted"].apply(
            lambda x: 1 if x == "<30" else 0
        )

        X_train = train.drop(columns=["readmitted"])
        y_train = train["readmitted"]

        X_test = test.drop(columns=["readmitted"])
        y_test = test["readmitted"]

        numerical_columns = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Convert sparse matrices to dense arrays if necessary
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        train_arr = np.column_stack((X_train, y_train.to_numpy()))
        test_arr = np.column_stack((X_test, y_test.to_numpy()))

        os.makedirs(self.config.root_dir, exist_ok=True)

        np.save(
            os.path.join(self.config.root_dir, "train.npy"),
            train_arr
        )

        np.save(
            os.path.join(self.config.root_dir, "test.npy"),
            test_arr
        )

        joblib.dump(preprocessor, self.config.preprocessor_path)

        logger.info("Preprocessor Saved")

        print("Transformation Completed")