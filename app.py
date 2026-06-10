from flask import Flask, render_template, request, Response
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("model/crop_disease_model.h5")

# ================= LOAD CLASS LABELS =================
with open("model/class_indices.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# ================= SOLUTION LOGIC =================
def get_solution(disease_name):
    name = disease_name.lower()

    if "healthy" in name:
        return "🌱 Plant is healthy. Maintain proper watering and sunlight."

    elif "bacterial" in name:
        return "🦠 Use copper-based bactericide. Remove infected leaves."

    elif "early_blight" in name:
        return "🍂 Apply Mancozeb or Chlorothalonil fungicide."

    elif "late_blight" in name:
        return "💧 Use Metalaxyl fungicide. Avoid excess moisture."

    elif "leaf_mold" in name:
        return "🍃 Reduce humidity and apply sulfur fungicide."

    elif "septoria" in name:
        return "🧴 Remove infected leaves and use fungicide spray."

    elif "spider_mites" in name:
        return "🐞 Use Neem oil or insecticidal soap."

    elif "target_spot" in name:
        return "🎯 Apply Chlorothalonil fungicide."

    elif "mosaic_virus" in name:
        return "⚠️ No cure. Remove infected plant immediately."

    elif "yellowleaf" in name or "curl_virus" in name:
        return "🪰 Control whiteflies using insecticide."

    else:
        return "⚠️ General treatment: Use fungicide and maintain proper care."

# ================= UPLOAD FOLDER =================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')

# ================= IMAGE PREDICTION =================
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)

    result = class_names[predicted_class]

    # Format name nicely
    display_name = result.replace("_", " ").replace("__", " - ")

    # Get solution dynamically
    solution = get_solution(result)

    return render_template(
        'result.html',
        prediction=display_name,
        confidence=round(confidence, 2),
        solution=solution,
        image_path=file_path
    )

# ================= WEBCAM =================
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        img = cv2.resize(frame, (224,224))
        img = np.expand_dims(img, axis=0) / 255.0

        predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        result = class_names[predicted_class]
        display_name = result.replace("_", " ").replace("__", " - ")

        label = f"{display_name} ({confidence:.1f}%)"

        cv2.putText(frame, label, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True)