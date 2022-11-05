import logging
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def deserialize_model(path: str) -> Pipeline:
    try:
        with open(path, "rb") as fin:
            pipeline = pickle.load(fin)
            logger.info(f"Loaded pipeline: {pipeline}")
            return pipeline
    except FileNotFoundError as err:
        logger.error(err)
        sys.exit(0)


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)
    logger.info(f"predictions.shape is {predictions.shape}")
    return predictions
