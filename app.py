import streamlit as st
import tensorflow as tf
import numpy as np

# Function to preprocess the image


def preprocess_image(image, target_size):
    image = tf.image.resize(image, target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

# Define the model architecture


def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load the trained weights


def load_weights(model, weights_path):
    model.load_weights(weights_path)


# Define target image size
target_size = (32, 32)

st.title('Image Classification App')

# Upload image through Streamlit
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = tf.keras.preprocessing.image.load_img(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image, target_size)

    # Create and compile the model
    model = create_model((target_size[0], target_size[1], 3))
    load_weights(model, 'cnn.weights.h5')

    # Make predictions
    predictions = model.predict(processed_image)
    binary_prediction = (predictions >= 0.5).astype(int)

    # Display the prediction result
    if binary_prediction[0][0] == 0:
        st.write("The image is classified as 'fake'.")
    else:
        st.write("The image is classified as 'real'.")
