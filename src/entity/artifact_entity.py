from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str


@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    val_report_file_path:str


@dataclass
class DataTransformationArtifact:
    transformed_train_feature_file_path: str
    transformed_test_feature_file_path: str
    transformed_train_target_file_path: str
    transformed_test_target_file_path: str
    transformed_object_file_path: str


@dataclass
class ClassificationMetricArtifact:
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    roc_auc_score: float


@ dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    metric_artifact: str
    mlflow_run_id: Optional[str] = None
    """
    ID of the MLflow run started during training. ModelEvaluation resumes
    this exact run to log the accept/reject decision, so one MLflow run
    represents the full train -> evaluate lifecycle for a candidate model.
    """


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    changed_accuracy: float
    blob_name: str
    trained_model_path: str


@dataclass
class ModelPusherArtifact:
    container_name: str
    blob_name: str