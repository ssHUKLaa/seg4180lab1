import io

import pytest
from PIL import Image

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home_returns_200(client):
    """GET / should return 200 and confirm the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"running" in response.data


def test_predict_no_file_returns_400(client):
    """POST /predict with no file attached should return 400."""
    response = client.post("/predict")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file uploaded"


def test_predict_with_image_returns_mask(client):
    """POST /predict with a valid image should return mask_png_base64 and
    house_coverage_pct.  The mock model returns all-zero logits so sigmoid
    gives 0.5 everywhere — threshold > 0.5 means mask is all background."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        data={"file": (buf, "aerial.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "mask_png_base64" in data
    assert "house_coverage_pct" in data
    assert isinstance(data["house_coverage_pct"], float)
