import sys

from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
from src.entity.azure_estimator import AzureEstimator
from src.exception import MyException
from src.logger import logging


class ModelPusher:
    """
    Pushes the accepted model to Azure Blob Storage.
    """

    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact
    ):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys)


    def initiate_model_pusher(self) -> ModelPusherArtifact:

        try:

            logging.info("Entered Model Pusher Component.")

            ##############################################################
            # Check if model is accepted
            ##############################################################

            if not self.model_evaluation_artifact.is_model_accepted:

                logging.info(
                    "Candidate model was rejected. Skipping model push."
                )

                return ModelPusherArtifact(
                    container_name=self.model_pusher_config.container_name,
                    blob_name=self.model_pusher_config.blob_name
                )

            ##############################################################
            # Upload model
            ##############################################################

            estimator = AzureEstimator(
                container_name=self.model_pusher_config.container_name,
                blob_name=self.model_pusher_config.blob_name
            )

            estimator.save_model(
                local_model_path=self.model_evaluation_artifact.trained_model_path,
                remove=False
            )

            logging.info("Model uploaded successfully to Azure Blob Storage.")

            ##############################################################
            # Artifact
            ##############################################################

            model_pusher_artifact = ModelPusherArtifact(
                container_name=self.model_pusher_config.container_name,
                blob_name=self.model_pusher_config.blob_name
            )

            logging.info("Exited Model Pusher Component.")

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)