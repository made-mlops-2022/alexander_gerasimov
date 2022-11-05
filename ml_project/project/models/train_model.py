import logging
import pickle
from typing import Dict, Optional, Union, NoReturn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from project.entities.train_params import TrainParams

logger = logging.getLogger(__name__)

ClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainParams
) -> ClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            solver="liblinear", random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    logger.info("Model successfully fitted.")
    return model


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }
    logger.info(f"Metrics are: {metrics}")
    return metrics


def serialize_model(
    model: ClassificationModel,
    path: str,
    transformer: Optional[ColumnTransformer] = None,
) -> NoReturn:
    pipeline = Pipeline(
        (
            [
                ("transformer", transformer),
                ("model", model),
            ]
        )
    )
    with open(path, "wb") as fout:
        pickle.dump(pipeline, fout)
    logger.info(f"Pipeline saved to {path}")
    return path
