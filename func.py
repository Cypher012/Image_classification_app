import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd


def load_model():
    with open("./input/classification_classes_ILSVRC2012.txt", "r") as f:
        image_net_frames = f.read().split("\n")
        class_names = [name.split(",")[0]  for name in image_net_frames]
        class_names = class_names[:-1]

        config_file = "./models/DenseNet_121.prototxt"
        model_file = "./models/DenseNet_121.caffemodel"

        model = cv2.dnn.readNet(model_file, config_file, 'Caffe')
        return class_names, model
    
def classify(img, class_names, model):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    blob = cv2.dnn.blobFromImage(img, 0.017, (224, 224), (104, 117, 123))
    model.setInput(blob)
    outputs = model.forward()
    final_outputs = outputs[0]
    final_outputs = final_outputs.reshape(1000,1)
    label_id = np.argmax(final_outputs)
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    ravel = np.ravel(probs)
    top_predictions_id = np.argsort(ravel)[-5:][::-1]
    top_predictions_value = list(ravel[top_predictions_id])
    top_predictions_value = [float(f"{(num.item() * 100):.3f}") for num in sorted(top_predictions_value, reverse=True)]
    top_predictions_outcome = []
    top_predictions = []
    for id in top_predictions_id:
        top_predictions_outcome.append(class_names[id])
    for outname, prob in zip(top_predictions_outcome, top_predictions_value):
        top_predictions.append((outname, prob))
    return top_predictions
      
def classify_ui(image, predictions):
    outname, final_prob = predictions[0]
    st.image(image)
    st.header("Algorithm Predicts:")
    st.subheader(f"It's a '{outname.capitalize()}' picture")

    st.header("Probabilty:")
    st.subheader(f"{final_prob}%")

    st.header("Top 5 predictions: ")

    data = []
    for prediction, probablity in predictions:
        data.append({"Prediction": prediction.capitalize(), "Probability (%)": probablity})
    
    df = pd.DataFrame(data, index=range(1, len(data) +1))
    st.dataframe(df)