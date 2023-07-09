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

# Define the labels
labels = {
    (1, 0, 0): 'Non-Invasive',
    (0, 1, 0): 'Invasive',
    (0, 0, 1): 'Ostracod'
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image using TensorFlow
    resized_image = tf.image.resize_with_crop_or_pad(
        tf.keras.preprocessing.image.img_to_array(image),
        target_size[0],
        target_size[1]
    )

    # Normalize the image for deep learning
    normalized_image = (resized_image - 127.5) / 127.5

    # Add an extra dimension to match the model input shape
    processed_image = tf.expand_dims(normalized_image, axis=0)

    return processed_image

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the model
    predictions = model.predict(processed_image)

    # Get the rounded prediction
    rounded_prediction = np.round(predictions[0])

    # Get the corresponding label
    label = labels[tuple(rounded_prediction)]

    return label

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
        predicted_label = predict_image(image)

        # Display the predicted label
        st.write("Predicted Label:")
        st.write(predicted_label)

# Run the app
if __name__ == "__main__":
    main()
