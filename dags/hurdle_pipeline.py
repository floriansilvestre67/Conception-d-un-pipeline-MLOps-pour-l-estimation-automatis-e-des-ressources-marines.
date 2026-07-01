"""
dags/hurdle_pipeline.py
────────────────────────
DAG Airflow 2.9 — Pipeline Hurdle pour la biomasse acoustique d'anchois.

Tâches :
  preprocess       → Lecture, nettoyage, encodage cyclique, PowerTransformer
  optimize         → Export des hyperparamètres (code Optuna commenté)
  train_classifier → HistGradientBoostingClassifier + StandardScaler
  train_regressor  → HistGradientBoostingRegressor + StandardScaler
  evaluate         → Métriques + importance des variables
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

PIPELINE_DIR  = "/opt/airflow/pipeline"
SHARED_DATA   = "/shared/data"
SHARED_MODELS = "/shared/models"
RAW_DATA_PATH = "/data/raw/NASC_anchois.xls"

default_args = {
    "owner":       "airflow",
    "retries":     1,
    "retry_delay": timedelta(minutes=2),
}


def _run_script(script_name: str):
    env = os.environ.copy()
    env["SHARED_DATA_PATH"]   = SHARED_DATA
    env["SHARED_MODELS_PATH"] = SHARED_MODELS
    env["RAW_DATA_PATH"]      = RAW_DATA_PATH

    result = subprocess.run(
        [sys.executable, f"{PIPELINE_DIR}/{script_name}"],
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} a échoué (code {result.returncode})")


def run_preprocess():       _run_script("preprocess.py")
def run_optimize():         _run_script("optimize.py")
def run_train_classifier(): _run_script("train_classifier.py")
def run_train_regressor():  _run_script("train_regressor.py")
def run_evaluate():         _run_script("evaluate.py")


with DAG(
    dag_id="hurdle_pipeline",
    description="Pipeline Hurdle — Biomasse acoustique d'anchois",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["anchois", "hurdle", "mlops"],
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocess,
    )

    optimize = PythonOperator(
        task_id="optimize",
        python_callable=run_optimize,
    )

    train_classifier = PythonOperator(
        task_id="train_classifier",
        python_callable=run_train_classifier,
    )

    train_regressor = PythonOperator(
        task_id="train_regressor",
        python_callable=run_train_regressor,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=run_evaluate,
    )

    preprocess >> optimize >> train_classifier >> train_regressor >> evaluate
