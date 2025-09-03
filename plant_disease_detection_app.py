
import tensorflow as tf
import numpy as np
from PIL import Image
import gradio as gr
import requests
from io import BytesIO
import cv2

# Load your model (you'll need to upload trained_modela.keras to your space)
model = tf.keras.models.load_model('trained_modela.keras')

class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']



def predict_disease(image):
    """
    Predict plant disease from uploaded image using same preprocessing as your working cv2 method
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path) 
            
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert PIL image to array like load_img method
        #image = image.convert("RGB")  # Ensure 3 channels
        image = tf.keras.preprocessing.image.load_img(temp_path,target_size=(128, 128))
        #image = image.resize((128, 128))  # Resize
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch

        # Predict
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = prediction[0][result_index]
        disease_name = class_name[result_index]

        return f"Disease: {disease_name}\nConfidence: {confidence:.2%}"

    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload Plant Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Plant Disease Detection API",
    description="Upload an image of a plant leaf to detect diseases",
    examples=[
        # You can add example images here
    ]
)

if __name__ == "__main__":
    iface.launch()