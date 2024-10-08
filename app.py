import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to create the label mapping
def create_label_mapping(species_list):
    bird_species_dirs = sorted(species_list)
    label_mapping = {idx: species_dir.replace('_images', '') for idx, species_dir in enumerate(bird_species_dirs)}
    return label_mapping

# Load the trained model using tf.saved_model.load
model = tf.saved_model.load('saved_model/')

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
    predictions = model(img, training=False)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    values = list(predictions)[0] #list[array([[0.8843 , 0.02489, 0.0909 ]], dtype=float16)]
    threshold = .85
    if values[predicted_class] > threshold:
        # Map the predicted class to the bird species
        predicted_species = label_mapping.get(predicted_class, "Unknown Species")

        # Display the prediction
        st.markdown(f"## Predicted Bird Species: {predicted_species}")
    else:
        st.markdown("## Predicted Bird Species: otherwise")


