import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

# Load variables from .env (no-op if the file doesn't exist, e.g. in CI)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CKPT_PATH = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(
            torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"WARNING: No checkpoint at '{CKPT_PATH}'. Run train.py first.")
    model.to(DEVICE)
    model.eval()
    return model


model = _load_model()

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


@app.route("/")
def home():
    return "House Segmentation API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        orig_w, orig_h = image.size

        img_t = _transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_t)["out"]  # (1, 1, H, W)
            prob = torch.sigmoid(output)[0, 0]  # (H, W)
            mask = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Resize mask back to original image dimensions
        mask_img = Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST)

        # Encode mask as base64 PNG for JSON transport
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        house_pct = float(np.array(mask_img).mean()) / 255.0 * 100
        return jsonify({
            "mask_png_base64": mask_b64,
            "house_coverage_pct": round(house_pct, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
