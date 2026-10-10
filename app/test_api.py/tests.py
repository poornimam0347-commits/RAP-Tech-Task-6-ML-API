def test_predict():
    response = client.post(
        "/predict",
        json={"features": [0.0] * 30}
    )

    assert response.status_code == 200
