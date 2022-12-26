from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor


HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
HOST_SPLITTED_DATA_PATH = "/data/splitted/{{ ds }}"
HOST_MODELS_PATH = "/data/models/{{ ds }}"
AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "dag_train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7),
) as dag:
    sensor_train_data = FileSensor(
        task_id="sensor-train-data",
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/data.csv",
        poke_interval=30,
    )

    sensor_train_target = FileSensor(
        task_id="sensor-train-target",
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/target.csv",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input_dir {HOST_RAW_DATA_PATH} --output_dir {HOST_PROCESSED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess-train-data",
        do_xcom_push=False,
        auto_remove=True,
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input_dir {HOST_PROCESSED_DATA_PATH} --output_dir {HOST_SPLITTED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-split-train-data",
        do_xcom_push=False,
        auto_remove=True,
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {HOST_SPLITTED_DATA_PATH} --models_dir {HOST_MODELS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        auto_remove=True,
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input_dir {HOST_SPLITTED_DATA_PATH} --models_dir {HOST_MODELS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-validate-model",
        do_xcom_push=False,
        auto_remove=True,
    )

    [sensor_train_data, sensor_train_target] >> preprocess >> split >> train >> validate
