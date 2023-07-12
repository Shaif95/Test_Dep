import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the custom function
def get_f1(y_true, y_pred):
    return 0  # Blank function
# Load the model with custom_objects argument
model = tf.keras.models.load_model('single_image_model.h5', custom_objects={'get_f1': get_f1})

# Define the target image size
target_size = (40, 40)


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image using TensorFlow
    resized_image = tf.image.resize_with_crop_or_pad(
        tf.keras.preprocessing.image.img_to_array(image),
        target_size[0],
        target_size[1]
    )

    # Normalize the image for deep learning
    normalized_image = resized_image / 255

    # Add an extra dimension to match the model input shape
    processed_image = tf.expand_dims(normalized_image, axis=0)

    return processed_image

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the model
    predictions = model.predict(processed_image)
    y = np.round(predictions)

    res = []
    for i in y:
        if(i[0]==1):
            res.append("Non-Invasive")
        elif(i[1]==1):
            res.append("Invasive")
        else:
            res.append("Ostracod")

    # Get the corresponding label
    label = res

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

        # Resize the image for display
        resized_image = image.resize((20, 20))

        # Display the resized image
        st.image(resized_image, caption="Resized Image")

        # Process and predict the image
        predicted_label = predict_image(image)

        # Display the predicted label
        st.write("Predicted Label:")
        st.write(predicted_label)

# Run the app
if __name__ == "__main__":
    main()
