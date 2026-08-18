import os
import subprocess
from typing import Any, Dict, Optional

import mlflow

from src.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.logger import logging

_experiment_initialized = False


def init_experiment() -> None:
    global _experiment_initialized
    if _experiment_initialized:
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        _experiment_initialized = True
        logging.info(f"MLflow initialized. uri={MLFLOW_TRACKING_URI}, experiment={MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        logging.error(f"MLflow init failed: {e}")


def start_or_resume_run(run_id: Optional[str] = None, run_name: Optional[str] = None) -> Optional[str]:
    init_experiment()
    try:
        active = mlflow.active_run()

        # If a run is already active in this process (e.g. trainer left it
        # open and evaluation is running right after, same process), just
        # reuse it instead of calling start_run() again.
        if active is not None:
            if run_id is None or active.info.run_id == run_id:
                logging.info(f"MLflow run already active, reusing. run_id={active.info.run_id}")
                return active.info.run_id
            else:
                # a different run is active — close it before switching
                mlflow.end_run()

        if run_id:
            active_run = mlflow.start_run(run_id=run_id)
        else:
            active_run = mlflow.start_run(run_name=run_name)

        logging.info(f"MLflow run active. run_id={active_run.info.run_id}")
        return active_run.info.run_id

    except Exception as e:
        logging.error(f"MLflow start_or_resume_run failed: {e}")
        return None


def log_params(params: Dict[str, Any]) -> None:
    try:
        mlflow.log_params(params)
    except Exception as e:
        logging.error(f"MLflow log_params failed: {e}")


def log_metrics(metrics: Dict[str, float]) -> None:
    try:
        mlflow.log_metrics(metrics)
    except Exception as e:
        logging.error(f"MLflow log_metrics failed: {e}")


def set_tags(tags: Dict[str, Any]) -> None:
    try:
        mlflow.set_tags(tags)
    except Exception as e:
        logging.error(f"MLflow set_tags failed: {e}")


def log_xgb_model(model, artifact_path: str = "model") -> None:
    try:
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
    except Exception as e:
        logging.error(f"MLflow log_xgb_model failed: {e}")


def get_git_commit_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def end_run() -> None:
    try:
        mlflow.end_run()
    except Exception as e:
        logging.error(f"MLflow end_run failed: {e}")