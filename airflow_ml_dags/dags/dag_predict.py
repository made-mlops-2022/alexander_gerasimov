from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor


HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
HOST_PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 2,
    "retry_delay": timedelta(minutes=0.5),
}

with DAG(
    "dag_predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    sensor_inference_data = FileSensor(
        task_id="sensor-inference-data",
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/data.csv",
        poke_interval=30,
    )

    sensor_model = FileSensor(
        task_id="sensor-model",
        filepath="/opt/airflow/data/models/{{ var.value.model }}/model.pkl",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input_dir {HOST_RAW_DATA_PATH} --output_dir {HOST_PROCESSED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess-inference-data",
        do_xcom_push=False,
        auto_remove=True,
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input_dir {HOST_PROCESSED_DATA_PATH} --output_dir {HOST_PREDICTIONS_PATH} "
        "--models_dir /data/models/{{ var.value.model }}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        auto_remove=True,
    )

    [sensor_inference_data, sensor_model] >> preprocess >> predict
