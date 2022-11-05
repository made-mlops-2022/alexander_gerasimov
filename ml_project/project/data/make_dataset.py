import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from project.entities import SplitParams

logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    logger.info(f"data.shape is {data.shape}")
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplitParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    logger.info(f"train_data.shape is {train_data.shape}")
    logger.info(f"val_data.shape is {val_data.shape}")
    return train_data, val_data
