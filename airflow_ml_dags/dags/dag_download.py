from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 3,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    "dag_download",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        auto_remove=True,
    )

    download
