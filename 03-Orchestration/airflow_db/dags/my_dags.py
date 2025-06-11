from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import requests
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ml_pipeline.homework import set_experiment, run_data_prep, train_model, register_model_to_mlflow
import logging

mlflow.set_tracking_uri("http://mlflow-ui:5000")
mlflow.set_experiment("nyc-taxi-model")


# DAG default arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="full_ml_pipeline",
    description="Read, preprocess, train and register model via MLflow",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["mlflow", "mlops", "full_pipeline"],
) as dag:

    RAW_PATH = "/opt/airflow/data"
    DATA_OUTPUT_PATH = "/opt/airflow/data_output"
    MODEL_NAME = "linear_regression"

    prep_data = PythonOperator(
        task_id="prep_data",
        python_callable=run_data_prep,
        op_kwargs={
            "raw_data_path": RAW_PATH,
            "dest_path": DATA_OUTPUT_PATH,
            "dataset": "yellow"
        }
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={"data_path": DATA_OUTPUT_PATH},
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_to_mlflow,
        op_kwargs={
            "run_id": "{{ ti.xcom_pull(task_ids='train_model') }}",
            "model_name": MODEL_NAME
        }
    )

    prep_data >> train >> register_model
