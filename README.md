# AI Image Colorization 🎨🖤➡️🌈

An AI-powered Deep Learning web application that converts grayscale (black-and-white) images into realistic colorized images using Computer Vision and Deep Learning techniques.

Built using Flask, TensorFlow, OpenCV, and deployed on Render.

---

# 🚀 Features

* Upload grayscale images
* AI automatically colorizes the image
* Deep Learning based image processing
* Responsive modern UI
* Same-size input and output images
* Real-time image prediction
* Flask web deployment
* Render cloud hosting

---

# 🧠 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* OpenCV
* NumPy
* HTML5
* CSS3
* Render Deployment

---

# 📂 Project Structure

```text
image_colorization_project/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── Procfile
│
├── models/
│   └── best_colorization_model.keras
│
├── static/
│   ├── uploads/
│   └── outputs/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

# 📊 Dataset

The model was trained on grayscale and colored face image datasets downloaded from Kaggle.

Dataset preprocessing included:

* Image resizing
* RGB to LAB conversion
* L-channel extraction
* Normalization
* Data augmentation

---

# ⚙️ Model Training

The project uses a CNN-based image colorization architecture.

Training Process:

1. Images converted from RGB → LAB
2. L channel used as input
3. AB channels predicted by CNN
4. Model trained using TensorFlow/Keras

Techniques Used:

* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint

---

# 🌐 Deployment

The Flask application is deployed using Render.

Deployment includes:

* Gunicorn server
* TensorFlow model loading
* OpenCV image processing
* Cloud hosting

---

# 📥 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/image-colorization-project.git
```

Move into the project directory:

```bash
cd image-colorization-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

---

# 🖼️ Usage

1. Open the web application
2. Upload a grayscale image
3. Click on **Colorize Image**
4. View the AI-generated colorized output

# Live link
https://image-colorization-w27r.onrender.com/predict
