from tensorflow.keras.models import load_model
model = load_model("models/best_colorization_model.keras")
model.summary()
print("Input shape:", model.input_shape)
