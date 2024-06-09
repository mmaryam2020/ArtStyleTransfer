# main.py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import os
import gradio as gr
from config import FILE_PATH

# this is a repository for trained machine learning models
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_and_preprocess_image(image):
	max_dim = 512
	image = Image.fromarray(image)
	# save file temporarily
	image.save(FILE_PATH)

	img = tf.io.read_file(FILE_PATH)
	# remove file
	os.remove(FILE_PATH)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]

	return img


# Function to perform style transfer
def style_transfer(content_image, style_image):
    try:
        content_image = load_and_preprocess_image(content_image)
        style_image = load_and_preprocess_image(style_image)
        generated_image = model(tf.constant(content_image), tf.constant(style_image))[0]
        generated_image = np.array(generated_image)
        generated_image = np.reshape(generated_image, generated_image.shape[1:])
        return  generated_image, " "
    except Exception as e:
        error_msg = f"Error during style transfer: {e}"
        print(error_msg)
        return None, error_msg


# Create Gradio interface
demo = gr.Interface(
    fn=lambda content_image, style_image: style_transfer(content_image, style_image),
    inputs=[
        gr.Image(label="Upload Content Image"),
        gr.Image(label="Upload Style Image")
    ],
    outputs=[
        gr.Image(label="Transformed Image"),
        gr.Textbox(label="Error Message")
    ],
    title="Image Style Transformer",
    description="Upload a content image and a style image to generate a new image with the style applied to the content."
)



if __name__ == "__main__":
    demo.launch(share=True)

