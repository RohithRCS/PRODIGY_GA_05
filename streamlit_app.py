import os
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow_hub as hub

# Set up TensorFlow Hub to load compressed models
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load and preprocess the image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Streamlit UI components
st.title("Image Stylization with TensorFlow Hub")

# Upload content and style images
content_file = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Load and display the content and style images
    content_image = load_img(content_file)
    style_image = load_img(style_file)

    st.image(tensor_to_image(content_image), caption='Content Image', use_column_width=True)
    st.image(tensor_to_image(style_image), caption='Style Image', use_column_width=True)

    # Load the model from TensorFlow Hub
    st.write("Stylizing image...")
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Apply style transfer
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert tensor to image and display
    result_image = tensor_to_image(stylized_image)
    st.image(result_image, caption='Stylized Image', use_column_width=True)

    # Option to download the stylized image
    st.download_button(
        label="Download Stylized Image",
        data=result_image.tobytes(),
        file_name="stylized_image.png",
        mime="image/png"
    )
