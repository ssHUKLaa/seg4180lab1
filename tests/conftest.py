import os
from unittest.mock import MagicMock, patch

import torch

# Must be set before app.py is imported
os.environ["FLASK_SECRET_KEY"] = "ci-test-secret-key"
os.environ["MODEL_CHECKPOINT"] = "nonexistent_checkpoint.pth"  # triggers warning, skips load

# Mock the model constructor so tests never build the real DeepLabV3 network.
# The mock is callable and returns {"out": zeros} — same shape as real output.
_mock_model = MagicMock()
_mock_model.return_value = {"out": torch.zeros(1, 1, 512, 512)}

_deeplabv3_patcher = patch(
    "torchvision.models.segmentation.deeplabv3_resnet50",
    return_value=_mock_model,
)
_deeplabv3_patcher.start()
