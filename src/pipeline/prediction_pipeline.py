import sys
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from pandas import DataFrame

from src.entity.azure_estimator import AzureEstimator
from src.entity.config_entity import VehiclePredictorConfig


class VehicleData:
    """
    Represents a single customer input for prediction.
    """

    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Vehicle_Age,
        Vehicle_Damage,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
    ):

        try: 
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage

        except Exception as e:
            raise MyException(e, sys) from e
        

    def get_data_as_dataframe(self)-> DataFrame:

        data = {
            "Gender": [self.Gender],
            "Age": [self.Age],
            "Driving_License": [self.Driving_License],
            "Region_Code": [self.Region_Code],
            "Previously_Insured": [self.Previously_Insured],
            "Vehicle_Age": [self.Vehicle_Age],
            "Vehicle_Damage": [self.Vehicle_Damage],
            "Annual_Premium": [self.Annual_Premium],
            "Policy_Sales_Channel": [self.Policy_Sales_Channel],
            "Vintage": [self.Vintage],
        }

        return pd.DataFrame(data)
    



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
    
