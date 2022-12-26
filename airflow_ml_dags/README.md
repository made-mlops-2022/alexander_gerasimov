# Airflow ML-dags

## Installation

```
git clone https://github.com/made-mlops-2022/alexander_gerasimov.git
cd airflow_ml_dags
```

## Usage

```
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
```

- To open the interface go to http://localhost:8080/ (use `login = admin` and `password = admin`)

## Run linter

```
flake8 --max-line-length=119
```

