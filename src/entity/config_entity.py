from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_file: str
    train_file: str
    test_file: str

from dataclasses import dataclass


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    STATUS_FILE: str

from dataclasses import dataclass

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str
    preprocessor_path: str

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str
    model_path: str

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    model_path: str
    test_data_path: str
    metrics_file_name: str
    classification_report_file: str