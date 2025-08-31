
import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "trained_model.keras"
IMG_SIZE = (128, 128)

# ======================
# LOAD MODEL
# ======================
model = load_model(MODEL_PATH)

# Load class names
try:
    validation_set = tf.keras.utils.image_dataset_from_directory(
        'valid',
        labels="inferred",
        label_mode="categorical",
        batch_size=32,
        image_size=IMG_SIZE,
        shuffle=False
    )
    class_names = validation_set.class_names
except:
    class_names = ["class_0", "class_1", "class_2"]  # Replace with your classes

print("Loaded Classes:", class_names)

def predict_headless():
    """Completely headless prediction"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Warming up camera...")
    time.sleep(2)
    
    print("Capturing frame in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture frame")
        return
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, IMG_SIZE)
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    # Save result image
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Prediction: {class_names[predicted_idx]} (Confidence: {confidence:.2f})")
    plt.axis("off")
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Predicted Class: {class_names[predicted_idx]}")
    print(f"Confidence: {confidence:.2f}")
    print("Result saved as 'prediction_result.png'")
    
    # Print all probabilities
    print("\nAll class probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"{class_names[i]}: {prob:.4f}")

if __name__ == "__main__":
    predict_headless()

    