from pydantic import BaseModel


class HeartData(BaseModel):
    id: int = 0
    age: int = 30
    sex: int = 1
    cp: int = 2
    trestbps: int = 110
    chol: int = 236
    fbs: int = 0
    restecg: int = 2
    thalach: int = 128
    exang: int = 1
    oldpeak: float = 0
    slope: int = 1
    ca: int = 0
    thal: int = 1
