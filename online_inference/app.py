import os
import sys
import pickle
import time
from typing import List, Optional
import logging

from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd

from src.entities import HeartData, HeartResponse
from src.validate import check_validity

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)

pipeline: Optional[Pipeline] = None
start_time = time.time()
app = FastAPI()


@app.get("/")
def main():
    return HTTPException(status_code=200, detail="Let's start.")


@app.on_event("startup")
def model_loading():
    time.sleep(10)
    path = os.getenv("PATH_TO_MODEL", default="model.pkl")
    if path is None:
        error = f"Path to model {path} is None"
        logger.error(error)
        raise RuntimeError(error)
    global pipeline
    with open(path, "rb") as f:
        pipeline = pickle.load(f)


@app.get("/status")
def status():
    global start_time
    if time.time() - start_time > 60:
        raise RuntimeError
    return HTTPException(status_code=200, detail="Pipeline is ready.")


@app.api_route("/predict", response_model=List[HeartResponse], methods=["GET", "POST"])
def predict(request: List[HeartData]):
    for req in request:
        valid, error = check_validity(req)
        if not valid:
            raise HTTPException(status_code=400, detail=error)
    data = pd.DataFrame(req.__dict__ for req in request)
    idxs = [int(x) for x in data.id]
    predicts = pipeline.predict(data.drop("id", axis=1))
    return [
        HeartResponse(id=idx, condition=int(predict))
        for idx, predict in zip(idxs, predicts)
    ]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
