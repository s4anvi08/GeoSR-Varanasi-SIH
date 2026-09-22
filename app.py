import base64
import io
import os

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from model import GeoSR
from utils import (pil_to_tensor, tensor_to_pil, image_bytes_to_pil,
                    pil_to_bytes, compute_metrics, bicubic_baseline)

app = Flask(__name__)
CORS(app)  # allow the React dev server to call this API

CKPT_PATH = os.environ.get("GEOSR_CKPT", "./checkpoints/geosr.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_scale = 4


def load_model():
    """Lazy-loads the trained checkpoint. Falls back to an untrained
    (randomly initialized) model so the API still runs before you've
    trained on your own images -- useful for wiring up the frontend first."""
    global _model, _scale
    if _model is not None:
        return _model

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        _scale = ckpt.get("scale", 4)
        m = GeoSR(scale=_scale).to(DEVICE)
        m.load_state_dict(ckpt["state_dict"])
        print(f"Loaded trained checkpoint from {CKPT_PATH} (scale={_scale})")
    else:
        print(f"WARNING: no checkpoint at {CKPT_PATH} -- run train.py first. "
              f"Serving an UNTRAINED model for now (outputs will look wrong).")
        m = GeoSR(scale=_scale).to(DEVICE)

    m.eval()
    _model = m
    return _model


def image_to_data_url(img: Image.Image) -> str:
    b = pil_to_bytes(img, fmt="PNG")
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "checkpoint_loaded": os.path.exists(CKPT_PATH),
        "scale": _scale,
    })


@app.route("/api/infer", methods=["POST"])
def infer():
    """
    multipart/form-data:
      lr_image  - required, the medium-resolution input tile
      hr_image  - optional, a true high-res reference tile for metrics
    """
    if "lr_image" not in request.files:
        return jsonify({"error": "missing lr_image file"}), 400

    model = load_model()

    lr_bytes = request.files["lr_image"].read()
    lr_img = image_bytes_to_pil(lr_bytes)

    with torch.no_grad():
        lr_t = pil_to_tensor(lr_img).to(DEVICE)
        sr_t = model(lr_t)
    sr_img = tensor_to_pil(sr_t)

    baseline_img = bicubic_baseline(lr_img, _scale)

    result = {
        "sr_image": image_to_data_url(sr_img),
        "bicubic_baseline_image": image_to_data_url(baseline_img),
        "lr_image": image_to_data_url(lr_img),
        "output_size": sr_img.size,
        "scale": _scale,
    }

    if "hr_image" in request.files:
        hr_bytes = request.files["hr_image"].read()
        hr_img = image_bytes_to_pil(hr_bytes)
        result["metrics_sr"] = compute_metrics(sr_img, hr_img)
        result["metrics_bicubic"] = compute_metrics(baseline_img, hr_img)

    return jsonify(result)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
