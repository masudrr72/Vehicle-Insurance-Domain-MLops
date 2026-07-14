import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split


from src.entity.config_entity import DataIngestionConfig
from src.data_access.vehicle_data import VehicleData
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion initialized")
        except Exception as e:
            raise MyException(e, sys)
        
    
    def export_data_into_feature_store(self)->DataFrame:
        """
        Export MongoDB data to the feature store.
        """
        try:
            logging.info("Exporting data from MongoDB")
            my_data = VehicleData()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            
            if dataframe.empty:
                logging.error(f"The exported DataFrame is empty. Shape: {dataframe.shape}")
           
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok= True)
            logging.info(f"Saving exported data into: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        
        except Exception as e:
            raise MyException(e, sys)
        


    def split_data_as_train_test(self,dataframe: DataFrame) -> None:
        """
        Split the dataset into train and test sets.
        """

        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state= self.data_ingestion_config.random_state,
                stratify= dataframe[self.data_ingestion_config.target_column]
            )

            logging.info("Performed train test split on the dataframe")
            logging.info(f"Train shape: {train_set.shape}")
            logging.info(f"Test shape: {test_set.shape}")
            logging.info(f"Train target distribution:\n{train_set[self.data_ingestion_config.target_column].value_counts(normalize=True)}")
            logging.info(f"Test target distribution:\n{test_set[self.data_ingestion_config.target_column].value_counts(normalize=True)}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header = True)

            logging.info("Exported train and test file")
        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Execute the data ingestion pipeline.
        """

        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from MongoDB")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")
            logging.info("Exited initiate_data_ingestion method of DataIngestion Class")

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys)
        
    
