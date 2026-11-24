import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "static/uploads"
ALLOWED = {"png", "jpg", "jpeg", "webp"}
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = None
model_error = None

def load_model():
    global model, model_error
    if model is not None:
        return model
    if not Path(MODEL_PATH).exists():
        model_error = f"Model file not found: {MODEL_PATH}"
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model_error = None
        return model
    except Exception as exc:
        model_error = str(exc)
        return None

@app.route("/")
def index():
    return render_template("index.html", model_ready=Path(MODEL_PATH).exists())

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify(error="Please select an image."), 400
    file = request.files["image"]
    if not file or not file.filename:
        return jsonify(error="Please select an image."), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED:
        return jsonify(error="Use JPG, JPEG, PNG or WEBP images."), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    saved_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(path)
    image_url = url_for("static", filename=f"uploads/{saved_name}")

    detector = load_model()
    if detector is None:
        return jsonify(
            error="The website is running, but the trained AI model is not installed. Add your trained YOLO weights as best.pt (or set MODEL_PATH), then restart the app.",
            image_url=image_url,
            model_ready=False,
            model_detail=model_error
        ), 503

    try:
        results = detector.predict(source=path, conf=0.25, verbose=False)
        result = results[0]
        detections = []
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "label": str(names.get(cls_id, f"class_{cls_id}")),
                "confidence": round(float(box.conf[0].item()) * 100, 1),
                "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()]
            })
        plotted = result.plot()
        import cv2
        marked_name = f"marked_{saved_name.rsplit('.', 1)[0]}.jpg"
        marked_path = os.path.join(app.config["UPLOAD_FOLDER"], marked_name)
        cv2.imwrite(marked_path, plotted)
        return jsonify(
            image_url=image_url,
            result_url=url_for("static", filename=f"uploads/{marked_name}"),
            detections=detections,
            count=len(detections),
            model_ready=True
        )
    except Exception as exc:
        return jsonify(error=f"Detection failed: {exc}", image_url=image_url), 500

@app.get("/health")
def health():
    return jsonify(status="ok", model_file_exists=Path(MODEL_PATH).exists())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
