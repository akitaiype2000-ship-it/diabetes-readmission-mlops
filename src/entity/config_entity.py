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