import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        try:

            logger.info("Reading dataset")

            df = pd.read_csv(self.config.source_file)

            os.makedirs(self.config.root_dir, exist_ok=True)

            logger.info("Splitting train and test dataset")

            train, test = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            print(repr(self.config.train_file))
            print(repr(self.config.test_file))
            train.to_csv(
                self.config.train_file,
                index=False
            )
            test.to_csv(
                self.config.test_file,
                index=False
            )

            logger.info("Data ingestion completed")

            return (
                self.config.train_file,
                self.config.test_file
            )

        except Exception as e:
            raise CustomException(e, sys)