import pandas as pd
import requests
import json


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv").drop("condition", axis=1)
    data["id"] = range(len(data))
    request_data = data.to_dict(orient="records")
    print("Request example(first sample):")
    print(request_data[0])
    response = requests.post(
        "http://0.0.0.0:8000/predict", json.dumps(request_data)
    )
    print(f"Response status: {response.status_code}")
    print("Response example(first sample):")
    print(response.json()[0])
