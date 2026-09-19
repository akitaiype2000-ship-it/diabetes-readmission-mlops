import os
import pandas as pd

from src.logger import logger
from src.entity.config_entity import DataValidationConfig


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self):

        os.makedirs(self.config.root_dir, exist_ok=True)

        train = pd.read_csv("artifacts/data_ingestion/train.csv")

        report = []

        if train.empty:
            report.append("Dataset is empty")
        else:
            report.append("Dataset is not empty")

        duplicates = train.duplicated().sum()
        report.append(f"Duplicate rows: {duplicates}")

        missing = train.isnull().sum().sum()
        report.append(f"Total missing values: {missing}")

        question_marks = (train == "?").sum().sum()
        report.append(f"Total '?' values: {question_marks}")

        if "readmitted" in train.columns:
            report.append("Target column found")
        else:
            report.append("Target column NOT found")

        with open(self.config.STATUS_FILE, "w") as f:
            f.write("\n".join(report))

        logger.info("Data Validation Completed")