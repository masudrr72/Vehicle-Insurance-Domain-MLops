import sys
from src.logger import logging
from src.exception import MyException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import DataValidationConfig


from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataValidationArtifact




class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """ 
        try:
            logging.info("Entered the start_data_ingestion method")
            logging.info("Getting data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and test set from MongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
        

    def start_data_validation(self, data_ingestin_artifact: DataIngestionArtifact)-> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(data_ingesetion_artifact=data_ingestin_artifact,
                                             data_validation_config= self.data_validation_config
                                             )
            
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        




    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestin_artifact=data_ingestion_artifact)
        
        except Exception as e:
            raise MyException(e, sys)
