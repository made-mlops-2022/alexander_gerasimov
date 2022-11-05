from typing import NoReturn, Union
import numpy as np
import pandas as pd
import pytest
from project.models.encoders import SqrtTransformer

x = np.array([[42, 22, 33], [-1, -2, -3], [0.0, 0.0, 0.0]])
expected_result = x.copy()
expected_result -= np.min(expected_result, axis=0, keepdims=True)
expected_result = np.sqrt(expected_result)


@pytest.mark.parametrize(
    "test_input, expected", [(x, expected_result), (pd.DataFrame(x), expected_result)]
)
def test_SqrtTransformer_works_as_expected(
    test_input: np.ndarray, expected: Union[np.ndarray, pd.DataFrame]
) -> NoReturn:
    transformer = SqrtTransformer()
    transformed = transformer.fit_transform(test_input)
    if isinstance(expected, pd.DataFrame):
        assert (
            transformed.tolist() == expected.values.tolist()
        ), "SqrtTransformer failed to procees pd.Dataframe."
    else:
        assert (
            transformed.tolist() == expected.tolist()
        ), "SqrtTransformer failed to procees np.ndarray."
