from pandas import DataFrame

from src.entity.azure_estimator import AzureEstimator
from src.entity.config_entity import VehiclePredictorConfig
from src.exception import MyException
from src.logger import logging

import sys


class PredictionPipeline:
    """
    Loads the latest production model from Azure Blob Storage
    and serves predictions.
    """

    def __init__(self):
        try:
            self.config = VehiclePredictorConfig()

            self.estimator = AzureEstimator(
                container_name=self.config.container_name,
                blob_name=self.config.blob_name,
            )

            logging.info("Loading production model from Azure Blob Storage.")
            self.model = self.estimator.load_model()
            logging.info("Production model loaded successfully.")

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> int:
        """
        Predict customer response.
        
        Returns:
            0 -> Not Interested
            1 -> Interested
        """
        try:
            prediction = self.model.predict(dataframe)

            return int(prediction[0])

        except Exception as e:
            raise MyException(e, sys) from e

    def predict_proba(self, dataframe: DataFrame) -> float:
        """
        Predict probability of customer purchasing insurance.

        Returns:
            Probability of class 1.
        """
        try:
            probability = self.model.predict_proba(dataframe)

            return float(probability[0][1])

        except Exception as e:
            raise MyException(e, sys) from e