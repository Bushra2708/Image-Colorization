# =========================================================
# IMPORT LIBRARIES
# =========================================================

from flask import Flask, render_template, request

import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model

# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# FOLDERS
# =========================================================

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model("models/best_colorization_model.keras")

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

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    # -----------------------------------------------------
    # SAVE UPLOADED IMAGE
    # -----------------------------------------------------

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(upload_path)

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    img = cv2.imread(upload_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for model
    resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # -----------------------------------------------------
    # RGB -> LAB
    # -----------------------------------------------------

    lab = cv2.cvtColor(
        resized_img.astype(np.float32) / 255.0,
        cv2.COLOR_RGB2LAB
    )

    # Extract L channel
    L = lab[:, :, 0]

    # Normalize
    L_input = L / 100.0

    # Reshape
    L_input = L_input.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # -----------------------------------------------------
    # PREDICT AB CHANNELS
    # -----------------------------------------------------

    pred_AB = model.predict(L_input, verbose=0)[0]

    # Denormalize
    pred_AB = pred_AB * 128

    # -----------------------------------------------------
    # RECONSTRUCT LAB IMAGE
    # -----------------------------------------------------

    colorized_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))

    colorized_lab[:, :, 0] = L
    colorized_lab[:, :, 1:] = pred_AB

    # LAB -> RGB
    colorized_rgb = cv2.cvtColor(
        colorized_lab.astype(np.float32),
        cv2.COLOR_LAB2RGB
    )

    # Clip values
    colorized_rgb = np.clip(colorized_rgb, 0, 1)

    # Convert to uint8
    output_img = (colorized_rgb * 255).astype(np.uint8)

    # RGB -> BGR
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    # Ensure same size
    output_img = cv2.resize(output_img, (128, 128))

    # -----------------------------------------------------
    # SAVE OUTPUT IMAGE
    # -----------------------------------------------------

    output_filename = "output_" + file.filename

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    cv2.imwrite(output_path, output_img)

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return render_template(
        "index.html",
        uploaded_image=upload_path,
        output_image=output_path
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)