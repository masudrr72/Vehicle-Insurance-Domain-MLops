import sys
from pandas import DataFrame

from src.cloud_storage.azure_storage import AzureStorageService
from src.entity.estimator import MyModel
from src.exception import MyException


class AzureEstimator:
    """
    Handles loading, saving and prediction using the production model
    stored in Azure Blob Storage.
    """

    def __init__(self,container_name: str, blob_name: str):
        self.container_name = container_name
        self.blob_name = blob_name
        self.storage = AzureStorageService()
        self.loaded_model: MyModel = None

    def is_model_present(self) -> bool:
        """
        Checks whether the production model exists.
        """
        try:
            return self.storage.blob_exists(container_name=self.container_name, blob_name=self.blob_name)

        except Exception as e:
            raise MyException(e, sys)

    def load_model(self) -> MyModel:
        """
        Downloads and returns the production model.
        """

        try:

            return self.storage.download_model(container_name=self.container_name,blob_name=self.blob_name)

        except Exception as e:
            raise MyException(e, sys)

    def save_model(self,local_model_path: str,remove: bool = False) -> None:
        """
        Upload local model.pkl to Azure Blob Storage.
        """

        try:

            self.storage.upload_file(
                source_file_path=local_model_path,
                container_name=self.container_name,
                blob_name=self.blob_name,
                remove=remove
            )

        except Exception as e:
            raise MyException(e, sys)

    def predict( self,dataframe: DataFrame):
        """
        Predict using the production model.
        """

        try:

            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            return self.loaded_model.predict(dataframe)

        except Exception as e:
            raise MyException(e, sys)
        


    def predict_proba(self,dataframe: DataFrame):
        """
        Predict probabilities using the production model.
        """

        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            return self.loaded_model.predict_proba(dataframe)

        except Exception as e:
            raise MyException(e, sys)