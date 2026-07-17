import os
from datetime import datetime


# for MongoDB connection:
DATABASE_NAME = "vehicle-project"
COLLECTION_NAME =  "vehicle_project_data"
MONGODB_URL_KEY =  "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Response"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# Data ingestion

DATA_INGESTION_COLLECTION_NAME: str = "vehicle_project_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20
RANDOM_STATE: int = 42

# Data validation

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# Data transformation

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessor.pkl"

TRAIN_FEATURE_FILE_NAME: str = "X_train.npy"
TEST_FEATURE_FILE_NAME: str = "X_test.npy"

TRAIN_TARGET_FILE_NAME: str = "y_train.npy"
TEST_TARGET_FILE_NAME: str = "y_test.npy"


# Model training

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

SCALE_POS_WEIGHT: float = 7.16
N_ESTIMATORS: int= 300
LEARNING_RATE: float = 0.05
MAX_DEPTH: int = 7
MIN_CHILD_WEIGHT: int = 5
RANDOM_STATE: int = 42
SUBSAMPLE: float = 0.8
REG_LAMBDA: int = 3
REG_ALPHA: float= 0.1
GAMMA: float = 0.1
COLSAMPLE_BYTREE: float = 0.8
















