
import os
import urllib.request
import urllib.error
import json

import pytest


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


@pytest.mark.skipif(
    os.getenv("RUN_DOCKER_TESTS") != "1",
    reason="Docker smoke tests are opt-in",
)
def test_docker_prediction():
    payload = {
        "tenure_months": 6,
        "monthly_spend": 100,
        "support_tickets": 8,
    }

    request = urllib.request.Request(
        f"{API_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=10,
    ) as response:
        assert response.status == 200
        result = json.load(response)

    assert result["prediction"] == 1
    assert result["model_version"] == "v2"
    assert result["churn_probability"] == 0.9959
