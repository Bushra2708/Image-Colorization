# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# LIMIT FILE SIZE
# =========================================================

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# =========================================================
# MODEL PATH & GLOBAL LOAD
# =========================================================

MODEL_PATH = "models/best_colorization_model.keras"
print("Loading model globally...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 128

# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------------------------------------------------
        # CHECK FILE
        # -------------------------------------------------
        if "image" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        # -------------------------------------------------
        # READ IMAGE INTO MEMORY
        # -------------------------------------------------
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return "Invalid image uploaded", 400

        original_h, original_w = img.shape[:2]

        # -------------------------------------------------
        # REDUCE LARGE IMAGES (BUT KEEP HIGH-RES FOR QUALITY)
        # -------------------------------------------------
        MAX_DIM = 800
        if max(original_h, original_w) > MAX_DIM:
            scale = MAX_DIM / max(original_h, original_w)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            img = cv2.resize(img, (new_w, new_h))
        else:
            new_h, new_w = original_h, original_w

        # -------------------------------------------------
        # BGR -> RGB & EXTRACT HIGH-RES L CHANNEL
        # -------------------------------------------------
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        
        L_high_res = img_lab[:, :, 0]

        # -------------------------------------------------
        # RESIZE FOR MODEL & PREDICT
        # -------------------------------------------------
        resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        lab_resized = cv2.cvtColor(resized_img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        L_resized = lab_resized[:, :, 0]
        
        L_input = (L_resized / 100.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

        pred_AB = model.predict(L_input, verbose=0)[0]
        pred_AB = pred_AB * 128

        # -------------------------------------------------
        # UPSACLE COLOR & COMBINE WITH HIGH-RES L
        # -------------------------------------------------
        pred_AB_high_res = cv2.resize(pred_AB, (new_w, new_h))

        colorized_lab = np.zeros((new_h, new_w, 3))
        colorized_lab[:, :, 0] = L_high_res
        colorized_lab[:, :, 1:] = pred_AB_high_res

        # -------------------------------------------------
        # LAB -> RGB
        # -------------------------------------------------
        colorized_rgb = cv2.cvtColor(colorized_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        colorized_rgb = np.clip(colorized_rgb, 0, 1)
        
        output_img = (colorized_rgb * 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        # -------------------------------------------------
        # ENCODE IMAGES TO BASE64
        # -------------------------------------------------
        _, buffer_in = cv2.imencode('.jpg', img)
        uploaded_image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer_in).decode('utf-8')}"

        _, buffer_out = cv2.imencode('.jpg', output_img)
        output_image_data = f"data:image/jpeg;base64,{base64.b64encode(buffer_out).decode('utf-8')}"

        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------
        return render_template(
            "index.html",
            uploaded_image=uploaded_image_data,
            output_image=output_image_data
        )

    # =====================================================
    # ERROR HANDLING
    # =====================================================
    except Exception as e:
        return f"""
        <h2 style='color:red; text-align:center; margin-top:50px;'>
            Error Occurred
        </h2>
        <p style='text-align:center; color:white;'>
            {str(e)}
        </p>
        """

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)