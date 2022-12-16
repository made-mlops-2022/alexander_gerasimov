import pytest
from fastapi.testclient import TestClient
import json

from app import app
from src.entities import HeartData


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_status():
    with TestClient(app) as client:
        response = client.get("/status")
        assert 200 == response.status_code


def test_out_of_range_values():
    with TestClient(app) as client:
        data = HeartData()
        test_values = [-100, 300]
        for value in test_values:
            data.age = value
            response = client.post("/predict", data=json.dumps([data.__dict__]))
            assert 400 == response.status_code


def test_binary_values():
    with TestClient(app) as client:
        data = HeartData()
        test_values = [11, -1]
        for value in test_values:
            data.sex = value
            response = client.post("/predict", data=json.dumps([data.__dict__]))
            assert 400 == response.status_code


def test_wrong_data_type():
    with TestClient(app) as client:
        data = HeartData()
        test_values = ["forever young", ""]
        for value in test_values:
            data.age = value
            response = client.post("/predict", data=json.dumps([data.__dict__]))
            assert 422 == response.status_code


@pytest.fixture()
def test_data():
    data = [HeartData(id=0), HeartData(id=1)]
    return data


def test_predict(test_data):
    with TestClient(app) as client:
        response = client.post(
            "/predict", data=json.dumps([x.__dict__ for x in test_data])
        )
        assert 200 == response.status_code
        assert 0 == response.json()[0]["id"]
        assert response.json()[0]["condition"] in [0, 1]
        assert len(response.json()) == len(test_data)
