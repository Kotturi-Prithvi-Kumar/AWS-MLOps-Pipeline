
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_customer(client):
    payload = {
        "tenure_months": 6,
        "monthly_spend": 1500,
        "support_tickets": 6,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert result["model_version"] == "v2"


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure_months", -5),
        ("monthly_spend", -100),
        ("support_tickets", -1),
    ],
)
def test_predict_negative_values(client, field, value):
    payload = {
        "tenure_months": 6,
        "monthly_spend": 1500,
        "support_tickets": 6,
    }

    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_field(client):
    payload = {
        "tenure_months": 6,
        "monthly_spend": 1500,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_extra_field(client):
    payload = {
        "tenure_months": 6,
        "monthly_spend": 1500,
        "support_tickets": 6,
        "customer_name": "Test Customer",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
