import os
import csv
import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Flask アプリ作成
app = Flask(__name__)

# モデルのパス（環境変数 MODEL_PATH があればそれを優先）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt"))

# モデルをロード（存在しなければ None）
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"[INFO] Model loaded: {MODEL_PATH}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}. Starting without model.")
except Exception as e:
    print(f"[WARN] Failed to load model: {e}")
    model = None

# ログ用のフォルダとCSV
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

# index ページ
@app.route("/")
def index():
    return render_template("index.html")

# 検出API
@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    results = model(file.read())

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": xyxy
            })

    # CSV に保存
    if detections:
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for d in detections:
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    d["class_id"],
                    d["confidence"],
                    *d["bbox"]
                ])

    return jsonify(detections)

# 起動
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
