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
    re_image = tf.image.resize_with_crop_or_pad(
        tf.keras.preprocessing.image.img_to_array(image),40,40)

    # Normalize the image for deep learning
    re_image = re_image / 255.0
    print(re_image.shape)

    return re_image

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions using the model
    pred = model.predict(processed_image)
    y = np.argmax( pred )

    res = []
    for i in y:
        if(i==0):
            res.append("Non-Invasive")
        elif(i==1):
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
