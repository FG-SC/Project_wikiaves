import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Function to create the label mapping
def create_label_mapping(image_folder_path):
    bird_species_dirs = sorted(os.listdir(image_folder_path))
    label_mapping = {idx: species_dir.replace('_images', '') for idx, species_dir in enumerate(bird_species_dirs)}
    return label_mapping

# Load the trained model
model = tf.keras.models.load_model('') #'bird_species_classifier.keras'
#model = tf.keras.layers.TFSMLayer(filepath='/saved_model.pb')
# Path to the folder containing bird images organized in subfolders by species
# image_folder_path = 'images_folders'

# Generate the label mapping
label_mapping = create_label_mapping(['urutau', 'gaviao-real', 'coruja-buraqueira'])

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(256, 256)):
    img = load_img(image, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image to [0, 1] range
    return img

# Streamlit app
st.title("Bird Species Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    img = preprocess_image(uploaded_file)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    values = list(predictions)[0] #list[array([[0.8843 , 0.02489, 0.0909 ]], dtype=float16)]
    threshold = .75
    if values[predicted_class]>threshold:
        # Map the predicted class to the bird species
        predicted_species = label_mapping.get(predicted_class, "Unknown Species")

        # Display the prediction
        st.write(f"Predicted Bird Species: {predicted_species}")
    else:
        st.write(f"Predicted Bird Species: otherwise")

