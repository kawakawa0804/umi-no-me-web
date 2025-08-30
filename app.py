from flask import Flask, render_template, Response, request, jsonify
import cv2, base64, numpy as np, pandas as pd
import datetime, csv, pathlib, glob, os
from ultralytics import YOLO

# ────────────────────────── アプリ設定 ────────────────────────── #
app = Flask(__name__, static_folder='static')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

# ────────────────────────── 初期化 ────────────────────────── #
cam   = None
model = YOLO('models/best.pt')                 # 学習済み重み

BASE_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR  = BASE_DIR / 'logs'                   # /.../umi-no-me-web/logs
LOG_DIR.mkdir(exist_ok=True)                  # 無ければ作成

# ──────────────────── 共通ユーティリティ ─────────────────── #
def gen_frames():
    """カメラ映像を Motion-JPEG でストリーム"""
    global cam
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

def append_csv(rows):
    """
    検出結果 rows(List[dict]) を logs/detections_YYYYMMDD_HHMM.csv へ追記
    - time は YYYY-MM-DD HH:MM:SS
    """
    fn = LOG_DIR / f"detections_{datetime.datetime.now():%Y%m%d_%H%M}.csv"
    new = not fn.exists()
    header = ["time", "label", "conf", "x1", "y1", "x2", "y2"]
    with open(fn, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        for r in rows:
            # 秒までに揃える
            r["time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            w.writerow(r)

# ──────────────────────── ルーティング ────────────────────── #
@app.route('/')
def home():
    return render_template('index.html')

# ---------- Camera ---------- #
@app.route('/camera')
def camera_page():
    return render_template('camera.html')

@app.route('/camera-feed')
def camera_feed():
    global cam
    if cam is None:
        cam = cv2.VideoCapture(0)
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.post('/detect')
def detect():
    """
    JS から base64 で送られてくる画像を推論し、結果を JSON で返す
    """
    img_b64 = request.json['image'].split(',')[1]
    arr  = np.frombuffer(base64.b64decode(img_b64), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    res = model(frame, verbose=False)[0]
    detections = []
    for xyxy, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy]
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": model.names[int(cls)],
            "conf": round(float(conf), 3)
        })
    if detections:
        append_csv(detections)
    return jsonify(detections=detections)

# ---------- CSV Viewer ---------- #
@app.route('/csv')
def csv_page():
    return render_template('csv.html')

@app.route('/csv-data')
def csv_data():
    """
    logs 内のすべての CSV を結合して返す。
    """
    files = sorted(LOG_DIR.glob('detections_*.csv'))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            app.logger.warning(f"Skip {f.name}: {e}")

    if dfs:
        df = pd.concat(dfs, ignore_index=True, join="outer", sort=False)
        df.sort_values('time', ascending=False, inplace=True)
        coord_cols = ["x1", "y1", "x2", "y2"]
        df[coord_cols] = df[coord_cols].round(2)
        df.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
    else:
        df = pd.DataFrame(columns=["time","label","conf","x1","y1","x2","y2"])

    return jsonify(df.to_dict(orient='records'))

# ---------- 終了時にカメラ解放 ---------- #
@app.teardown_appcontext
def release_cam(_):
    global cam
    if cam is not None:
        cam.release()
        cam = None

# ────────────────────────── エントリーポイント ────────────────────────── #
if __name__ == '__main__':
    app.run(debug=True)
