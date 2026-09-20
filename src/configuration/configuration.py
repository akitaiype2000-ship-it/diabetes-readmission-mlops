import yaml

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from src.utils import create_directories

from src.entity.config_entity import ModelTrainerConfig
from src.entity.config_entity import ModelEvaluationConfig
class ConfigurationManager:

    def __init__(self):
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

    def get_data_ingestion_config(self):

        config = self.config["data_ingestion"]

        return DataIngestionConfig(
            root_dir=config["root_dir"],
            source_file=config["source_file"],
            train_file=config["train_file"],
            test_file=config["test_file"],
        )

    def get_data_validation_config(self):

        config = self.config["data_validation"]

        return DataValidationConfig(
            root_dir=config["root_dir"],
            STATUS_FILE=config["STATUS_FILE"],
        )

    def get_data_transformation_config(self):

        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        return DataTransformationConfig(
            root_dir=config["root_dir"],
            train_data_path=config["train_data_path"],
            test_data_path=config["test_data_path"],
            preprocessor_path=config["preprocessor_path"],
        )
    def get_model_trainer_config(self):

        config = self.config["model_trainer"]

        create_directories([config["root_dir"]])

        return ModelTrainerConfig(
        root_dir=config["root_dir"],
        train_data_path=config["train_data_path"],
        test_data_path=config["test_data_path"],
        model_path=config["model_path"],
    )
    def get_model_evaluation_config(self):

        config = self.config["model_evaluation"]

        create_directories([config["root_dir"]])

        return ModelEvaluationConfig(
    root_dir=config["root_dir"],
    model_path=config["model_path"],
    test_data_path=config["test_data_path"],
    metrics_file_name=config["metrics_file_name"],
    classification_report_file=config["classification_report_file"],
)