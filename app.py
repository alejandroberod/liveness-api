import cv2
import numpy as np
from flask import Flask, jsonify, request

from anti_spoof.predictor import AntiSpoofService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

service = AntiSpoofService()


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="missing_image", detail="Send the image in the 'image' form field"), 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="invalid_image"), 400

    result = service.predict(image)
    if result is None:
        return jsonify(error="no_face_detected"), 400

    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="image_too_large", detail="Max size is 10 MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
