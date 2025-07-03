import imghdr
import os
import shutil
import cv2
from keras.src.utils.image_utils import img_to_array
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import ModelBuilder as I
import zipfile
import time

# Inject custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Replace with your image URL or local path
image_url = "https://deepvisionai.in/img/fac.jpg"

# HTML/CSS to center and style the image
st.markdown(
    f"""
    <style>
    .image-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }}
    .responsive-img {{
        height: 200px;
        width: 400px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    </style>
    <div class="image-container">
        <img class="responsive-img" src="{image_url}" alt="Styled Image">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stApp {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit UI
st.title("I Spy with AI")
#st.subheader("Add Your Name Here")
st.write("Upload an image and let the model predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", ".zip"])
class_names = []

# Get class names from your training data (same order as before)
if st.button("Build Model"):
    with st.spinner("Processing..."):
        progress = st.progress(0)
        for i in range(50):
            time.sleep(0.02)
            progress.progress(i + 1)
    st.info("Building Your Model.....Please Wait")
    class_names = I.createModel(35) # Replace with your actual class names

    for i in range(50):
        time.sleep(0.02)
        progress.progress(i + 51)
    st.success("All Done!")

else:
    st.write("Click Here to Build Your Model")

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".zip"):
        data_dir = 'data/'
        extract_to = data_dir + str(os.path.splitext(uploaded_file.name)[0])
        # Create the directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        # Unzip the file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        for image_class in os.listdir(data_dir):
            if image_class == str(os.path.splitext(uploaded_file.name)[0]):
                imageNotFound = True
                for image in os.listdir(os.path.join(data_dir, image_class)):
                    image_path = os.path.join(data_dir, image_class, image)
                    try:
                        img = cv2.imread(image_path)
                        tip = imghdr.what(image_path)
                        if tip in image_exts and imageNotFound:
                            dst_folder = "test_images"
                            os.makedirs(dst_folder, exist_ok=True)
                            print("Image Path: " + str(image_path))
                            shutil.move(image_path,dst_folder)
                            imageNotFound = False
                            break
                    except Exception as e:
                        message = 'Issue with image {}'.format(image_path)
                        # os.remove(image_path)
    else:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        class_names = os.listdir("data")
        st.write(class_names)

        # Preprocess image
        img_height, img_width = 224, 224
        image = image.resize((img_width, img_height))
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Load the trained model
        model = tf.keras.models.load_model("model.h5")

        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions)
        st.write(predictions)
        st.write(score)
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")