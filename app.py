from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import pandas as pd
from func import *
 
with st.sidebar:
    st.title("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    st.text('OR')
    url = st.text_input("Enter Url")
    if uploaded_image:
        classify_button = st.button('Classify Image')

# Main content area

st.title("Image Classification App")

class_names, model = load_model()


if uploaded_image is not None:
    image = np.array(Image.open(uploaded_image))
    predictions = classify(image, class_names, model)
    if classify_button:
        classify_ui(image, predictions)

elif url != "":
    try:
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        predictions = classify(image, class_names, model)
        classify_ui(image, predictions)
    except MissingSchema as err:
        st.header("Invalid URL, Try Agaiin")
    except UnidentifiedImageError as err:
        st.header("URL has no Image, Try again")








