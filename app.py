import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

import streamlit as st


model = load_model("models/Image_classify.keras")
data_cat = [
  'apple',
  'banana',
  'beetroot',
  'bell pepper',
  'cabbage',
  'capsicum',
  'carrot',
  'cauliflower',
  'chilli pepper',
  'corn',
  'cucumber',
  'eggplant',
  'garlic',
  'ginger',
  'grapes',
  'jalepeno',
  'kiwi',
  'lemon',
  'lettuce',
  'mango',
  'onion',
  'orange',
  'paprika',
  'pear',
  'peas',
  'pineapple',
  'pomegranate',
  'potato',
  'raddish',
  'soy beans',
  'spinach',
  'sweetcorn',
  'sweetpotato',
  'tomato',
  'turnip',
  'watermelon'
]

# Width And Height Img Size Param
img_width = 180
img_height = 180

# Web Interface
st.header("Klasifikasi Gambar Buah Dan Sayur dari Model")
image_name = st.text_input("Enter Image Name", "Apple.jpg")
image = "image/"+image_name

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image)
st.write("Veg/Fruit in image is " + data_cat[np.argmax(score)])
st.write("With Accuracy of " + str(np.max(score)*100))



