import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.azure_estimator import AzureEstimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class EvaluateModelResponse:
    trained_model_roc_auc_score: float
    production_model_roc_auc_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    Compare Candidate Model with Production Model.
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)


    def get_production_model(self) -> Optional[AzureEstimator]:
        """
        Returns production model estimator if available.
        """

        try:

            estimator = AzureEstimator(
                container_name=self.model_eval_config.container_name,
                blob_name=self.model_eval_config.blob_name,
            )

            if estimator.is_model_present():
                return estimator

            return None

        except Exception as e:
            raise MyException(e, sys)


    def evaluate_model(self) -> EvaluateModelResponse:

        try:

            logging.info("Loading test dataset")

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path
            )

            X_test = test_df.drop(columns=[TARGET_COLUMN])

            y_test = test_df[TARGET_COLUMN]

            ###############################################################
            # Candidate Model
            ###############################################################

            logging.info("Loading candidate model")

            candidate_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            candidate_probability = candidate_model.predict_proba(X_test)[:, 1]

            candidate_score = roc_auc_score(
                y_test,
                candidate_probability
            )



            ###############################################################
            # Production Model
            ###############################################################

            production_model = self.get_production_model()

            if production_model is None:

                logging.info("No Production model found.")

                production_score = 0.0

            else:

                logging.info("Production model found.")

                production_probability = production_model.predict_proba(X_test)[:, 1]

                production_score = roc_auc_score(
                    y_test,
                    production_probability
                )

            ###############################################################
            # Decision
            ###############################################################

            difference = candidate_score - production_score

            is_model_accepted = (
                difference >
                self.model_eval_config.changed_threshold_score
            )

            response = EvaluateModelResponse(
                trained_model_roc_auc_score=candidate_score,
                production_model_roc_auc_score=production_score,
                is_model_accepted=is_model_accepted,
                difference=difference,
            )

            logging.info(f"Evaluation Result : {response}")

            return response

        except Exception as e:
            raise MyException(e, sys)


    def initiate_model_evaluation(
        self,
    ) -> ModelEvaluationArtifact:

        try:

            logging.info(
                "Starting Model Evaluation"
            )

            evaluation_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                changed_accuracy=evaluation_response.difference,
                blob_name=self.model_eval_config.blob_name,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )

            logging.info(
                "Model Evaluation Completed Successfully."
            )

            logging.info(
                f"{model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)