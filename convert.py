import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model("models/best_colorization_model.keras")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Saving TFLite model...")
with open("models/best_colorization_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Conversion complete!")
