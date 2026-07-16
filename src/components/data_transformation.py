import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import *



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)
        


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data
        """
        try:
            logging.info("Entered get_data_transformer_object method of DataTransformation class")
            # features loading
            numeric_features = self._schema_config['numeric_features']
            binary_features = self._schema_config['binary_features']
            ordinal_features = self._schema_config['ordinal_features']
            nominal_features = self._schema_config['nominal_features']

            # pipeline

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])


            binary_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])

            ordinal_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[
                    ["Female", "Male"],
                    ["< 1 Year", "1-2 Year", "> 2 Years"],
                    ["No", "Yes"]
                ]))  
            ])

            nominal_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output= False))   
            ])

            final_pipeline = ColumnTransformer([
                ("num", numeric_pipeline, numeric_features),
                ("bin", binary_pipeline, binary_features),
                ("ord", ordinal_pipeline, ordinal_features),
                ("nom", nominal_pipeline, nominal_features)
            ])


            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            logging.info("Input and Target cols defined for both train and test df.")
            feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            
            target_train_sr = train_df[TARGET_COLUMN]
            target_test_sr = test_df[TARGET_COLUMN]


            preprocessor = self.get_data_transformer_object()

            feature_train_arr = preprocessor.fit_transform(feature_train_df)
            feature_test_arr = preprocessor.transform(feature_test_df)

            target_train_arr = target_train_sr.to_numpy()
            target_test_arr = target_test_sr.to_numpy()


            logging.info("Saving transformation object and transformed files.")
            logging.info("Data transformation completed successfully")

            # saving preprocessing object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # saving transformed features
            save_numpy_array_data(self.data_transformation_config.transformed_train_feature_file_path, feature_train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_feature_file_path, feature_test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_train_target_file_path, target_train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_target_file_path, target_test_arr)



            return DataTransformationArtifact(   
                transformed_train_feature_file_path= self.data_transformation_config.transformed_train_feature_file_path,
                transformed_test_feature_file_path= self.data_transformation_config.transformed_test_feature_file_path,
                transformed_object_file_path= self.data_transformation_config.transformed_object_file_path,
                transformed_train_target_file_path= self.data_transformation_config.transformed_train_target_file_path,
                transformed_test_target_file_path= self.data_transformation_config.transformed_test_target_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
