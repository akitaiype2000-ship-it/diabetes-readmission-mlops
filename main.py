from src.configuration.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

# Load configuration
config = ConfigurationManager()

# -----------------------
# Data Ingestion
# -----------------------
data_ingestion_config = config.get_data_ingestion_config()

data_ingestion = DataIngestion(data_ingestion_config)

train_path, test_path = data_ingestion.initiate_data_ingestion()

print("Train file:", train_path)
print("Test file:", test_path)

# -----------------------
# Data Validation
# -----------------------
data_validation_config = config.get_data_validation_config()

data_validation = DataValidation(data_validation_config)

data_validation.validate()

print("Data Validation Completed Successfully!")


from src.components.data_transformation import DataTransformation
############################################
# DATA TRANSFORMATION
############################################

transformation_config = config.get_data_transformation_config()

data_transformation = DataTransformation(
    transformation_config
)

data_transformation.transform()

