from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()

with DAG(
    dag_id="gemstone_training_pipeline",                   # DAG Name
    default_args = {"retries": 2},                  # 
    description = "it is my training pipeline",
    schedule = "@daily",       # here you can test based on hour or mints but make sure here you container is up and running
    start_date = pendulum.datetime(2024, 8, 15, tz="UTC"),
    catchup = False,
    tags = ["machine_learning ", "regression", "mlops"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):       # a Dictionary is passed here by default when DAG triggers a component
        ti = kwargs["ti"]               # a KEY named "ti": (Task Instance) is present in the Dictionary
        train_data_path, test_data_path = training_pipeline.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":train_data_path, "test_data_path":test_data_path})   # 'xcom' : Cross-Communication between components : push & pull
                                                                                                                        # 'xcom_push': for PUSHING the Outputs to the Next Task component
    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact") 
        train_arr, test_arr = training_pipeline.start_data_transformation(data_ingestion_artifact["train_data_path"], data_ingestion_artifact["test_data_path"])    # 'xcom_pull': PULLING Outputs of the given Task ID 
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push("data_transformations_artifcat", {"train_arr":train_arr, "test_arr":test_arr})

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_training(train_arr, test_arr)
    
    # AWS / any cloud
    def push_data_to_s3(**kwargs):
        import os
        bucket_name = os.getenv("BUCKET_NAME")
        artifact_folder = "/app/artifacts"
        os.system(f"aws s3 sync {artifact_folder} s3:/{bucket_name}/artifact")
        
        

    # defining AirFlow OPERATORS for each Task:

    data_ingestion_task = PythonOperator(
        task_id = "data_ingestion",
        python_callable = data_ingestion
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    this task creates a train and test file.
    """
    )

    data_transform_task = PythonOperator(
        task_id = "data_transformation",
        python_callable = data_transformations
    )
    data_transform_task.doc_md = dedent(
        """\
    #### Transformation task
    this task performs the transformation
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable = model_trainer
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### model trainer task
    this task perform training
    """
    )
    
   
    push_data_to_s3_task = PythonOperator(
        task_id="push_data_to_s3",
        python_callable = push_data_to_s3
        )


# DAG Task Sequence:
data_ingestion_task >> data_transform_task >> model_trainer_task >> push_data_to_s3_task