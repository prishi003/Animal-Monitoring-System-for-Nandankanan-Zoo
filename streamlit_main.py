import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

# Load the trained model
model_path = 'my_model.keras'  # Update this path to the correct location
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(file_path):
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "Conservation Threat"
    else:
        result = "No Conservation Threat"
    return result

# Streamlit UI
st.title("Conservation Threat Prediction")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    # Save the uploaded file
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make prediction
    prediction = predict_image(file_path)

    st.write("Prediction:", prediction)
