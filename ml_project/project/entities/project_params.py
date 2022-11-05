from pathlib import Path
from dataclasses import dataclass


@dataclass()
class NumericalColumn:
    name: str
    mean: float
    std: float


@dataclass()
class CategoricalColumn:
    name: str
    nunique: int


APPLICATION_NAME = "ml_project"

NUMERICAL_COLUMNS = {
    "age": NumericalColumn("age", 54.366, 9.082),
    "trestbps": NumericalColumn("trestbps", 131.624, 17.538),
    "chol": NumericalColumn("chol", 246.264, 51.831),
    "thalach": NumericalColumn("thalach", 149.647, 22.905),
    "oldpeak": NumericalColumn("oldpeak", 1.04, 1.61),
}

CATEGORICAL_COLUMNS = {
    "sex": CategoricalColumn("sex", 2),
    "cp": CategoricalColumn("cp", 4),
    "fbs": CategoricalColumn("fbs", 2),
    "restecg": CategoricalColumn("restecg", 3),
    "exang": CategoricalColumn("exang", 2),
    "slope": CategoricalColumn("slope", 3),
    "ca": CategoricalColumn("ca", 5),
    "thal": CategoricalColumn("thal", 4),
}


LABEL_COLUMN = "condition"

ARTIFACT_DIR = Path("experiments")

DATA_DIR = Path("data")

PROCESSED_DIR = Path("data/processed")

REPORT_DIR = Path("reports")
