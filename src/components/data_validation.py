import json
import sys
import os

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file


class DataValidation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e,sys)
        


    def validate_number_of_features(self, dataframe: DataFrame)-> bool:
        """
        This method validates the number of columns
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns present: [{status}]")    
            return status
        except Exception as e:
            raise MyException(e, sys)
    


    def is_feature_exist(self, df:DataFrame) -> bool:
        """
        This method validates the existence of a numerical, categorical, ordinal and nominal  columns
        """
        try:
            dataframe_features =  df.columns
            missing_numeric_features = []
            missing_binary_features = []
            missing_ordinal_features = []
            missing_nominal_features = []

            for column in self._schema_config["numeric_features"]:
                if column not in dataframe_features:
                    missing_numeric_features.append(column)

            if len(missing_numeric_features)>0:
                logging.info(f"Missing Numerical feature: {missing_numeric_features}")

            for column in self._schema_config["binary_features"]:
                if column not in dataframe_features:
                    missing_binary_features.append(column)

            if len(missing_binary_features)>0:
                logging.info(f"Missing Binary feature: {missing_binary_features}")
    
            for column in self._schema_config["ordinal_features"]:
                if column not in dataframe_features:
                    missing_ordinal_features.append(column)

            if len(missing_ordinal_features)>0:
                logging.info(f"Missing Ordinal feature: {missing_ordinal_features}")

            for column in self._schema_config["nominal_features"]:
                if column not in dataframe_features:
                    missing_nominal_features.append(column)

            if len(missing_nominal_features)>0:
                logging.info(f"Missing Nominal feature: {missing_nominal_features}")


            return False if len(missing_numeric_features)>0 or len(missing_binary_features)>0 or len(missing_ordinal_features)>0 or len(missing_nominal_features)>0 else True

        except Exception as e:
            raise MyException(e, sys) from e
            


    @staticmethod
    def read_data(file_path)-> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    


    def initiate_data_validation(self)-> DataValidationArtifact:
        """
        This method initiates the data validation component for the pipeline
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.train_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            # validation of number of faetures
            status = self.validate_number_of_features(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe"
            else:
                logging.info(f"All required features are present in training dataframe: {status}")


            status = self.validate_number_of_features(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All required features are present in test dataframe: {status}")


            # validation the existence of features
            status = self.is_feature_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            else:
                logging.info(f"All numerical/categorical/ordinal/nominal features are present in training dataframe: {status}")

            status = self.is_feature_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All numerical/categorical/ordinal/nominal features are present in test dataframe: {status}")


            
            validation_status =  len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message = validation_error_msg,
                val_report_file_path= self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)


            # Save validation status and message to a JSON file      
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }
                
            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            
            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

