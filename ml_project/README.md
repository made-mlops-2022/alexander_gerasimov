ML_Project
==============================

Installation: 
~~~
python3 -m venv .venv
source .venv/bin/activate
pip install .
~~~
Usage:
~~~
Load EDA report
python3 project/reports/report.py -i data/heart_cleveland.csv -o reports/heart_cleveland

Train model
python3 project/train_pipeline.py

Predict with model
python3 project/predict_pipeline.py

Test:
pip install pytest pytest-cov
python -m pytest . -v --cov=project

Linter:
flake8 . --count  --statistics
~~~


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    ├── data               <- Data from third party sources
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Notebooks with EDA
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │    
    ├── configs            <- Configuration files for project modules
    │    
    ├── logs               <- Files with logs
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── tests              <- Code to test project modules and pipelines
    │
    ├── outputs            <- Hydra logs
    │
    ├── project            <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    │   ├── reports        <- Generated graphics and figures to be used in reporting
    │   │
    │   └── entities       <- Parameters for different project modules
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
