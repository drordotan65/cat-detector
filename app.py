from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return "ESP32 Cat Detector API is running."

@app.route("/detect", methods=["POST"])
def detect_cat():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    img_bytes = image_file.read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)[0]
    cat_detected = False
    confidence = 0.0

    if results.boxes is not None:
        detected_labels = [model.names[int(box.cls[0])] for box in results.boxes]
        print("Detected labels:", detected_labels)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "cat":
                cat_detected = True
                confidence = float(box.conf[0])
                break

    return jsonify({
        "cat_detected": cat_detected,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
