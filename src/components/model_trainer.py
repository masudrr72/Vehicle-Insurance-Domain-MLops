import sys
import time
from datetime import datetime
from typing import Tuple
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.utils.main_utils import load_numpy_array_data, save_object, load_object
from src.entity.estimator import MyModel
from src.utils import mlflow_utils


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array) -> Tuple[XGBClassifier, ClassificationMetricArtifact]:
        try:
            logging.info("Training XGBClassifier with specified parameters")
            model = XGBClassifier(
                scale_pos_weight=self.model_trainer_config._scale_pos_weight,
                n_estimators=self.model_trainer_config._n_estimators,
                learning_rate=self.model_trainer_config._learning_rate,
                max_depth=self.model_trainer_config._max_depth,
                min_child_weight=self.model_trainer_config._min_child_weight,
                subsample=self.model_trainer_config._subsample,
                reg_lambda=self.model_trainer_config._reg_lambda,
                reg_alpha=self.model_trainer_config._reg_alpha,
                gamma=self.model_trainer_config._gamma,
                colsample_bytree=self.model_trainer_config._colsample_bytree,
                random_state=self.model_trainer_config._random_state,
                eval_metric="logloss"
            )

            logging.info("Training model")
            training_start = time.time()
            model.fit(X_train, y_train)
            training_duration_seconds = time.time() - training_start
            logging.info("Model training complete")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(f"Accuracy : {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall   : {recall:.4f}")
            logging.info(f"F1 Score : {f1:.4f}")
            logging.info(f"ROC AUC  : {roc_auc:.4f}")

            metric_artifact = ClassificationMetricArtifact(
                accuracy_score = accuracy,
                precision_score = precision,
                recall_score = recall,
                f1_score = f1,
                roc_auc_score = roc_auc
            )

            ###########################################################
            # MLflow: params, metrics, imbalance/dataset tags, model
            # (must happen here, while `model` is still the raw
            # XGBClassifier - before it gets wrapped into MyModel)
            ###########################################################

            positive_count = int(np.sum(y_train == 1))
            negative_count = int(np.sum(y_train == 0))
            dataset_size = int(len(y_train) + len(y_test))

            mlflow_utils.log_params({
                "model_name": "XGBoost",
                "n_estimators": self.model_trainer_config._n_estimators,
                "learning_rate": self.model_trainer_config._learning_rate,
                "max_depth": self.model_trainer_config._max_depth,
                "min_child_weight": self.model_trainer_config._min_child_weight,
                "subsample": self.model_trainer_config._subsample,
                "colsample_bytree": self.model_trainer_config._colsample_bytree,
                "reg_lambda": self.model_trainer_config._reg_lambda,
                "reg_alpha": self.model_trainer_config._reg_alpha,
                "gamma": self.model_trainer_config._gamma,
                "scale_pos_weight": self.model_trainer_config._scale_pos_weight,
                "random_state": self.model_trainer_config._random_state,
            })

            mlflow_utils.set_tags({
                "class_balancing_strategy": "scale_pos_weight",
                "dataset_name": "vehicle-insurance",
                "model_stage": "candidate",
            })

            mlflow_utils.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "training_time_seconds": training_duration_seconds,
                "dataset_size": dataset_size,
                "positive_class_ratio": positive_count / len(y_train),
                "negative_class_ratio": negative_count / len(y_train),
            })

            mlflow_utils.log_xgb_model(model, artifact_path="model")

            
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            logging.info("Entered initiate_model_trainer method.")

            run_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow_run_id = mlflow_utils.start_or_resume_run(run_name=run_name)

            git_sha = mlflow_utils.get_git_commit_sha()
            if git_sha:
                mlflow_utils.set_tags({"git_commit": git_sha})

            X_train = load_numpy_array_data(
                file_path= self.data_transformation_artifact.transformed_train_feature_file_path
            )

            X_test = load_numpy_array_data(
                file_path= self.data_transformation_artifact.transformed_test_feature_file_path
            )

            y_train = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_train_target_file_path
            )

            y_test = load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_test_target_file_path
            )

            logging.info("Transformed train-test data loaded.")

            trained_model, metric_artifact = self.get_model_object_and_report(
                X_train,
                X_test,
                y_train,
                y_test
            )

            preprocessing_object = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            final_model = MyModel(
                preprocessing_object= preprocessing_object,
                trained_model_object=trained_model
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )

            logging.info("Trained model saved successfully.")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                mlflow_run_id=mlflow_run_id
            )

            logging.info("Exited initiate_model_trainer method.")

            return model_trainer_artifact

        except Exception as e:
            mlflow_utils.end_run()
            raise MyException(e, sys)
        






    