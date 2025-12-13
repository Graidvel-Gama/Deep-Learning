import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def load_model():
    try:
        model = tf.keras.models.load_model("best_model.h5")
        return model
    except:
        return FileNotFoundError("Model could not be found.")

model = load_model()

file = st.file_uploader("Upload an image", type = ["jpg", "jpeg", "png"])
st.write("File needs to be in PNG, JPG, or JPEG.")

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgresized = cv2.resize(imgrgb, (224, 224))
    processedimg = preprocess_input(imgresized)
    processedimg = np.expand_dims(processedimg, axis = 0)

    pred = model.predict(processedimg)[0][0]
    label = "Fresh" if pred < 0.5 else "Spoiled"

    st.image(imgrgb, caption="Uploaded Image", use_column_width=True)

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Raw prediction score: {pred:.3f}")
