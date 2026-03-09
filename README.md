# SEG4180 Lab 2 — House Segmentation API

This is my Lab 2 submission. It builds on Lab 1 by replacing the image classification model with a segmentation model that detects houses in aerial satellite imagery. I also added secrets management, a CI/CD pipeline, and trained the model on a real dataset.

## What it does

The Flask API takes an aerial image and returns a binary segmentation mask showing where houses are, along with the percentage of the image covered by houses.

---

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**2. Create your `.env` file** (copy from the example and fill in your values):
```bash
cp .env.example .env
```

The `.env` file needs:
```
FLASK_SECRET_KEY=your_secret_key_here
FLASK_DEBUG=false
PORT=5000
HF_TOKEN=your_huggingface_token
MODEL_CHECKPOINT=checkpoints/best_model.pth
SAM_CHECKPOINT=sam_vit_b_01ec64.pth
DATASET_CONFIG=full
```

You can generate a secret key with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Running the API

Make sure you have a trained checkpoint at `checkpoints/best_model.pth` (see Training below), then:

```bash
python app.py
```

The server starts at `http://localhost:5000`.

**Test it:**
```bash
# Home endpoint
curl http://localhost:5000/

# Predict on an image
curl -X POST http://localhost:5000/predict -F "file=@test_image.jpg"
```

Response looks like:
```json
{
  "house_coverage_pct": 23.4,
  "mask_png_base64": "iVBORw0KGgo..."
}
```

---

## Reproducing the full pipeline

### Step 1 — Download the SAM checkpoint
```bash
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b_01ec64.pth
```

### Step 2 — Prepare the dataset
This runs SAM on the HuggingFace satellite dataset to generate pixel masks. It took about 20 hours on my machine with a 4GB GPU. It supports pause/resume — type `pause`, `resume`, or `quit` while it's running.
```bash
python prepare_dataset.py
```
Output goes to `dataset/{train,val,test}/{images,masks}/`.

### Step 3 — Train the model
```bash
python train.py
```
Runs for 20 epochs and saves the best checkpoint to `checkpoints/best_model.pth`. Training curves are saved to `plots/training_curves.png`. Took about 15 hours on my GPU at 512×512.

If you want it faster, use a smaller image size:
```bash
python train.py --img-size 256 --batch-size 4
```

### Step 4 — Evaluate
```bash
python evaluate.py
```
Prints mean IoU and Dice score over the full test set, and saves a visualization grid to `plots/predictions.png`.

---

## Running the tests

```bash
pytest tests/ -v
```

The tests mock the model so they don't require a GPU or checkpoint. Should pass in a few seconds.

---

## CI/CD

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) runs automatically on every push:
1. Linting with flake8
2. Unit tests with pytest
3. Docker image build and push to Docker Hub (on pushes to `main` only)

Docker Hub secrets (`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`) need to be set in the GitHub repo settings under Secrets.

---

## Project structure

```
app.py                  Flask API
train.py                Model training script
evaluate.py             Evaluation and visualizations
prepare_dataset.py      SAM-based dataset preparation
tests/                  Unit tests
.github/workflows/      CI/CD pipeline
checkpoints/            Saved model weights (gitignored)
dataset/                Prepared image/mask pairs (gitignored)
plots/                  Training curves and prediction visualizations
.env.example            Template for environment variables
```
