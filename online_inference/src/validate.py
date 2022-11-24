from typing import Tuple, Union
from src.entities import HeartData


def check_validity(data: HeartData) -> Tuple[bool, str]:
    def check_binary(param_value: int, param_name: str):
        if param_value not in [0, 1]:
            raise ValueError(f"{param_name} has a non-binary value{param_value}.")

    def check_numeric(
        param_value: Union[int, float],
        l_bound: Union[int, float],
        u_bound: Union[int, float],
        param_name: str,
    ):
        if not (l_bound <= param_value <= u_bound):
            raise ValueError(
                f"{param_name} has value {param_value} which is out of [{l_bound}, {u_bound}] range."
            )
    try:
        check_numeric(data.age, 0, 100, "age")
        check_binary(data.sex, "sex")
        check_numeric(data.cp, 0, 3, "cp")
        check_numeric(data.trestbps, 100, 200, "trestbps")
        check_numeric(data.chol, 100, 450, "chol")
        check_binary(data.fbs, "fbs")
        check_numeric(data.restecg, 0, 2, "restecg")
        check_numeric(data.thalach, 70, 250, "thalach")
        check_binary(data.exang, "exang")
        check_numeric(data.oldpeak, 0, 10, "oldpeak")
        check_numeric(data.slope, 0, 2, "slope")
        check_numeric(data.ca, 0, 4, "ca")
        check_numeric(data.thal, 0, 3, "thal")
        return True, "OK"
    except ValueError as error:
        return False, str(error)
