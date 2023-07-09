import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the custom function
def get_f1(y_true, y_pred):
    return 0  # Blank function

# Load the model with custom_objects argument
model = tf.keras.models.load_model('my_model.h5', custom_objects={'get_f1': get_f1})

# Define the target image size
target_size = (40, 40)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image using PIL
    image = image.resize(target_size)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize the image for deep learning
    normalized_image = (image_array - 127.5) / 127.5

    # Add an extra dimension to match the model input shape
    processed_image = np.expand_dims(normalized_image, axis=0)

    return processed_image

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the model
    predictions = model.predict(processed_image)

    # Get the rounded prediction
    rounded_prediction = np.round(predictions[0])

    return rounded_prediction

# Streamlit app
def main():
    # Set the app title
    st.title("Image Recognition App")

    # Upload and display the image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image from the uploaded file
        image = Image.open(uploaded_file)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Process and predict the image
        rounded_prediction = predict_image(image)

        # Display the rounded prediction
        st.write("Rounded Prediction:")
        st.write(rounded_prediction)

# Run the app
if __name__ == "__main__":
    main()
