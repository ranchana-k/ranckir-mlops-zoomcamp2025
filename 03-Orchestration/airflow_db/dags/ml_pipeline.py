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

# MLflow setup
mlflow.set_tracking_uri("sqlite:////opt/airflow/models/mlflow.db")
mlflow.set_experiment("taxi-model")

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
LOCAL_PARQUET = "/opt/airflow/models/yellow_tripdata_2023-03.parquet"
LOCAL_GZIP = "/opt/airflow/models/yellow_tripdata_2023-03.parquet.gzip"

def download_parquet():
    r = requests.get(URL)
    with open(LOCAL_PARQUET, "wb") as f:
        f.write(r.content)
    print("✅ Downloaded Parquet")

def compress_parquet():
    df = pd.read_parquet(LOCAL_PARQUET, engine="pyarrow")
    df.to_parquet(LOCAL_GZIP, compression="gzip")
    print("📦 Compressed to GZIP")

def train_model():
    df = pd.read_parquet(LOCAL_GZIP)
    df = df.dropna(subset=["trip_distance", "fare_amount"])
    X = df[["trip_distance"]]
    y = df["fare_amount"]

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

    print(f"✅ Trained and logged model. RMSE: {rmse:.3f}")

with DAG(
    dag_id="taxi_pipeline_local",
    start_date=datetime(2024, 1, 1),
    schedule="@once",
    catchup=False
) as dag:

    t1 = PythonOperator(task_id="download", python_callable=download_parquet)
    t2 = PythonOperator(task_id="compress", python_callable=compress_parquet)
    t3 = PythonOperator(task_id="train_and_log", python_callable=train_model)

    t1 >> t2 >> t3
