import pandas as pd
import mlflow
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
import numpy as np
import os
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
pd.set_option('display.float_format', '{:.2f}'.format)


def set_experiment():
    # mlflow.set_tracking_uri("sqlite:///mlflow_test.db")
    # mlflow.set_experiment("nyc_taxi_hw3")
    # MLflow setup
    mlflow.set_tracking_uri("http://mlflow-ui:5000")
    mlflow.set_experiment("nyc-taxi-model")


# In[8]:


def read_dataframe(filename):
    df = pd.read_parquet(filename)
    print(f"{filename} has {len(df)} records.")
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.apply(lambda td: td.total_seconds()/60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID','DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


# In[9]:


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


# In[10]:


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


# In[11]:


def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    # print(raw_data_path)
    # Load parquet files
    logging.info("Start reading parquet files.")
    try: 
        df_train = read_dataframe(
            os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
        )
        print(f"df_train records: {len(df_train)}")
        df_val = read_dataframe(
            os.path.join(raw_data_path, f"{dataset}_tripdata_2023-04.parquet")
        )
        print(f"df_val records: {len(df_)}")
    except Exception as e:
        logging.ERROR(f"An error occured: {e}")
    # df_test = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
    # )
    logging.info(f"df_train records: {len(df_train)}")
    # Extract the target
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    # y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    # X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    # os.makedirs(dest_path, exist_ok=True)

    
    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    # dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


# In[41]:


def load_pickle(filename: str):
    print(filename)

    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
def train_model(data_path):
    
    with mlflow.start_run() as run:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        lr = LinearRegression()
        
        lr.fit(X_train, y_train)

      
        mlflow.log_param("intercept_", lr.intercept_)
        y_pred = lr.predict(X_val)
        # ตัวอย่างข้อมูลที่ใช้ infer
        input_example = X_train[:5]
        signature = infer_signature(X_train[:5], lr.predict(X_train[:5]))

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(sk_model=lr, artifact_path="artifacts",
                                 input_example=input_example,
                                 signature=signature)
        run_id = run.info.run_id
        
        
    return run_id
        


# In[24]:


def register_model_to_mlflow(run_id: str, model_name: str, artifact_path: str = "artifacts") -> str:
    """
    Register a model from a completed MLflow run to the MLflow Model Registry.

    Parameters:
        run_id (str): The run ID of the MLflow run.
        model_name (str): The name to register the model under in the MLflow Model Registry.
        artifact_path (str): The artifact path used when logging the model. Default is "artifacts".

    Returns:
        str: The model URI that was registered.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"

    mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"✅ Model registered: {model_name} from run_id={run_id}")
    return model_uri



