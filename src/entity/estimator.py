import numpy as np
import pandas as pd


class MyModel:
    """
    Wrapper class that combines the preprocessing object
    and the trained model.
    """

    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing and return predictions.
        """
        transformed_data = self.preprocessing_object.transform(dataframe)
        prediction = self.trained_model_object.predict(transformed_data)

        return prediction

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing and return prediction probabilities.
        """
        transformed_data = self.preprocessing_object.transform(dataframe)
        probability = self.trained_model_object.predict_proba(transformed_data)

        return probability