import os
import csv
import datetime
from flask import Flask, render_template, request, jsonify

# ===== Flask =====
app = Flask(__name__)

# ===== Paths & Env =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.pt"))

# Ultralytics が /opt/render/.config に書けない時の警告回避（/tmp を使わせる）
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ===== Lazy model holder =====
model = None

def ensure_model_loaded():
    """必要になったタイミングでのみモデルを読み込む（起動を軽くする）"""
    global model
    if model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}")
        return False
    try:
        # import を遅延させて起動を軽く
        from ultralytics import YOLO
        # torch import もここで（重いので遅延）
        try:
            import torch  # noqa: F401
            # CPU スレッド数を絞ってメモリ/CPU使用を抑える
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
        except Exception:
            pass
        m = YOLO(MODEL_PATH)
        model = m
        print(f"[INFO] Model loaded: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None
        return False

# ===== Logs =====
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "detections.csv")

# ===== Routes =====
@app.route("/")
def index():
    # テンプレートが無い場合でも 200 を返す簡易ビュー
    try:
        return render_template("index.html")
    except Exception:
        return "<h1>Umi no Me</h1><p>Server is running.</p>", 200

@app.route("/health")
def health():
    return "ok", 200

@app.route("/detect", methods=["POST"])
def detect():
    # モデルが無ければ 503（サービス未準備）
    if not ensure_model_loaded():
        return jsonify({"error": "Model not available"}), 503

    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image uploaded (use form field 'image' or 'file')"}), 400

    file = request.files.get("image") or request.files.get("file")
    # Ultralytics は bytes かパスを受け付ける
    from ultralytics.utils import ops  # noqa: F401  # 一部環境での初回遅延を前倒し
    results = model(file.read())

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            detections.append({"class_id": cls_id, "confidence": conf, "bbox": xyxy})

    # CSV に保存
    if detections:
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for d in detections:
                w.writerow([
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    d["class_id"], d["confidence"], *d["bbox"]
                ])

    return jsonify(detections)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
