# Bird Species Classifier with Streamlit

## Overview

This project showcases a bird species classification model trained to identify images of three bird species using a Convolutional Neural Network (CNN) implemented in TensorFlow. The model is deployed as a Streamlit web application, allowing users to upload an image and receive a prediction of the bird species. If the model isn't confident about the image being one of the bird species, it classifies it as "otherwise."

## Dataset

The dataset was scraped from the WikiAves website, a popular platform for birdwatchers in Brazil. Images were collected for the following three species:
- Urutau
- Gavião-real
- Coruja-buraqueira

Approximately 60 images per species were scraped, providing a balanced dataset for training.

### Image Preparation

- **Tools Used**: `numpy`, `pandas`, `requests`, `os`, `PIL`, `tf.keras.preprocessing.image`
- Images were processed and resized to 256x256 pixels.
- The images were normalized to ensure uniformity across the dataset.
- Image data augmentation techniques were employed to enhance model generalization.

## Model

The model is a Convolutional Neural Network (CNN) designed to classify images into one of the three bird species. Key features include:
- **Mixed Precision Training**: Utilized `tf.keras.mixed_precision` to train the model in `float16` format, optimizing memory usage and speeding up training.
- **One-Hot Encoding**: Implemented with `sklearn.preprocessing.OneHotEncoder` to facilitate the calculation of confusion matrix and ROC/AUC curves.

### Performance Evaluation

The model's performance was evaluated using:
- **Confusion Matrix**: To understand the model’s accuracy in classifying each species.
- **ROC/AUC Curve**: To assess the model's ability to distinguish between the bird species.
  
These metrics were crucial in refining the model and ensuring it performs reliably across different species.

## Application

The Streamlit app serves as the user interface for the model, allowing users to upload images and receive predictions in real-time. Key features include:
- **Threshold Classification**: A threshold is set to ensure predictions are made only when the model is confident. If the confidence score is below the threshold, the app outputs "otherwise," ensuring that non-bird images (e.g., a car) do not yield false positive results.
- **User Interface**: Simple and intuitive, designed for ease of use by both bird enthusiasts and general users.

## How It Works

1. **Upload an Image**: Users can upload a `.jpg` image to the app.
2. **Image Processing**: The uploaded image is preprocessed to match the input requirements of the model.
3. **Prediction**: The model predicts the bird species or classifies the image as "otherwise" based on the confidence threshold.
4. **Output**: The predicted species or the "otherwise" classification is displayed.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bird-species-classifier.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Conclusion

This project demonstrates the power of machine learning in wildlife conservation, offering an easy-to-use tool for identifying bird species. The use of mixed precision training, along with robust evaluation metrics, ensures that the model performs efficiently even on modest hardware, making it accessible to a wide audience.

Feel free to explore the app, provide feedback, or contribute to the project!
