import os
import tensorflow as tf
import streamlit as st
import numpy as np
import PIL.Image
import requests
from io import BytesIO
import tensorflow_hub as hub

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = st.secrets["TFHUB_MODEL_LOAD_FORMAT"]

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load and preprocess the image from URL
def load_img_from_url(url):
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    img = tf.convert_to_tensor(np.array(img), dtype=tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32)

    max_dim = 512
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Streamlit UI components
st.title("Image Stylization with TensorFlow Hub")

# Input URLs for content and style images
content_url = st.text_input("Enter the content image URL:")
style_url = st.text_input("Enter the style image URL:")

if content_url and style_url:
    # Load and display the content and style images
    content_image = load_img_from_url(content_url)
    style_image = load_img_from_url(style_url)


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
