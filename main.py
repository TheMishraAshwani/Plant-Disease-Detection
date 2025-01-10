pip install gdown
import gdown
import os

# URL of the dataset on Google Drive

dataset_url = 'https://drive.google.com/uc? id=10QeUB2F9P6nYwJzJIsBK9p5J1q1WDpxs

# Destination folder where the dataset will be downloaded
dataset_folder = 'dataset'

# Download the dataset
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

gdown.download(dataset_url, output=dataset_folder, quiet=False)



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up an ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_data',  # Path to the directory containing training images
    target_size=(512, 512),  # Resize images
    batch_size=32,
    class_mode='categorical')  # Change to 'binary' if it's a binary classification task

# Train the model
model.fit(train_generator, epochs=10)

# Save the model after training
model.save("my_trained_model.h5")

import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

@st.cache(allow_output_mutation=True)
def load_model(path):
    
    
    xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4,activation='softmax')
    ])

    
    inputs = tf.keras.Input(shape=(512, 512, 3))

    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)

    outputs = tf.keras.layers.average([densenet_output, xception_output])


    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    
    model.load_weights(path)
    
    return model



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


model = load_model(r"C:\Users\rahul\OneDrive\Desktop\Plant-Disease-Detection-main\model (1).h5")


st.title('Plant Diesease Detection')
st.write("Just Upload your Plant's Leaf Image and get predictions if the plant is healthy or not")


uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg"])


if uploaded_file != None:
    
    
    progress = st.text("Crunching Image")
    my_bar = st.progress(0)
    i = 0
    
    
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(Image.fromarray(
        np.array(image)).resize((700, 400), Image.ANTIALIAS)), width=None)
    my_bar.progress(i + 40)
    
    
    image = clean_image(image)
    
    
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 30)
    
   
    result = make_results(predictions, predictions_arr)
    
    
    my_bar.progress(i + 30)
    progress.empty()
    i = 0
    my_bar.empty()
    
    
    st.write(f"The plant {result['status']} with {result['prediction']} prediction.")

# model used = https://drive.google.com/file/d/10QeUB2F9P6nYwJzJIsBK9p5J1q1WDpxs/view?usp=drivesdk
