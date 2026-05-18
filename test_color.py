import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/best_colorization_model.keras")

IMG_SIZE = 128

img_path = "static/black-white-portrait-8658058.webp"
img = cv2.imread(img_path)

original_h, original_w = img.shape[:2]
MAX_DIM = 800
if max(original_h, original_w) > MAX_DIM:
    scale = MAX_DIM / max(original_h, original_w)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    img = cv2.resize(img, (new_w, new_h))
else:
    new_h, new_w = original_h, original_w

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)

L_high_res = img_lab[:, :, 0]

resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
lab_resized = cv2.cvtColor(resized_img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
L_resized = lab_resized[:, :, 0]

L_input = (L_resized / 100.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

pred_AB = model.predict(L_input, verbose=0)[0]
pred_AB = pred_AB * 128

pred_AB_high_res = cv2.resize(pred_AB, (new_w, new_h))

colorized_lab = np.zeros((new_h, new_w, 3))
colorized_lab[:, :, 0] = L_high_res
colorized_lab[:, :, 1:] = pred_AB_high_res

colorized_rgb = cv2.cvtColor(colorized_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
colorized_rgb = np.clip(colorized_rgb, 0, 1)

output_img = (colorized_rgb * 255).astype(np.uint8)
output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

cv2.imwrite("test_output_128.jpg", output_img)
print("Saved test_output_128.jpg")
