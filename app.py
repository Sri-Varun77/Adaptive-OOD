import io
from pathlib import Path

import torch
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError

from data_pipeline import get_transforms
from models import AuxiliaryDetector, FeatureExtractor, MODEL_NAMES, create_original_model, class_names_from_synsets

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
app.secret_key = "adaptive-ood-local-session-key"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ARTIFACTS = {}
LOAD_ERROR = None


def load_artifacts():
    global ARTIFACTS, LOAD_ERROR
    loaded = {}
    try:
        for name in MODEL_NAMES:
            original_path = OUTPUT_DIR / f"{name}_original.pt"
            auxiliary_path = OUTPUT_DIR / f"{name}_auxiliary.pt"
            if not original_path.exists() or not auxiliary_path.exists():
                raise FileNotFoundError(
                    f"Missing trained files for {name}. Run python train_two_models.py first."
                )
            original_checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
            auxiliary_checkpoint = torch.load(auxiliary_path, map_location="cpu", weights_only=False)
            model = create_original_model(name, pretrained=False)
            model.load_state_dict(original_checkpoint["model_state"])
            model.to(DEVICE).eval()
            detector = AuxiliaryDetector(auxiliary_checkpoint["feature_dim"])
            detector.load_state_dict(auxiliary_checkpoint["detector_state"])
            detector.to(DEVICE).eval()
            extractor = FeatureExtractor(model, name)
            loaded[name] = {
                "model": model,
                "detector": detector,
                "extractor": extractor,
                "mean": auxiliary_checkpoint["feature_mean"].to(DEVICE),
                "std": auxiliary_checkpoint["feature_std"].to(DEVICE),
                "threshold": float(auxiliary_checkpoint["threshold"]),
                "classes": original_checkpoint["classes"],
                "class_names": original_checkpoint["class_names"],
            }
        ARTIFACTS = loaded
        LOAD_ERROR = None
    except Exception as exc:
        for item in loaded.values():
            item["extractor"].close()
        ARTIFACTS = {}
        LOAD_ERROR = str(exc)


def run_prediction(image):
    tensor = get_transforms(False)(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    results = []
    with torch.no_grad():
        for name in MODEL_NAMES:
            artifact = ARTIFACTS[name]
            features, logits = artifact["extractor"](tensor)
            normalized_features = (features - artifact["mean"]) / artifact["std"]
            probability = torch.sigmoid(artifact["detector"](normalized_features))[0].item()
            class_probabilities = torch.softmax(logits, dim=1)[0]
            top_count = min(3, class_probabilities.numel())
            top_values, top_indices = torch.topk(class_probabilities, top_count)
            top_classes = []
            for confidence, class_index in zip(top_values.tolist(), top_indices.tolist()):
                top_classes.append({
                    "name": artifact["class_names"][class_index],
                    "confidence": confidence * 100.0,
                })
            results.append({
                "name": name,
                "decision": "YES — supported class" if probability >= artifact["threshold"] else "NO — outside supported classes",
                "is_id": probability >= artifact["threshold"],
                "id_score": probability * 100.0,
                "threshold": artifact["threshold"] * 100.0,
                "top_classes": top_classes,
            })
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        if not ARTIFACTS:
            flash(LOAD_ERROR or "Trained models are not available. Run python train_two_models.py first.", "error")
            return redirect(url_for("index"))
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            flash("Choose an image first.", "error")
            return redirect(url_for("index"))
        try:
            image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            results = run_prediction(image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            flash(f"The selected file is not a valid image: {exc}", "error")
    return render_template("index.html", results=results, ready=bool(ARTIFACTS), load_error=LOAD_ERROR)


load_artifacts()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
