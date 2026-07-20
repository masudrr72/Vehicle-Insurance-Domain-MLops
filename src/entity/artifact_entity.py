from dataclasses import dataclass


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