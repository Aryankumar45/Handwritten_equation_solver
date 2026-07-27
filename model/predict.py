import os
import sys
import cv2
import numpy as np
# import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
# from keras.utils import image_dataset_from_directory

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocess import preprocess_equation


# os.system("cls")
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "model.keras")
model = load_model(path, compile=False)
model.trainable = False


value = {'cap A' : "A", 'cap Y' : "Y", 'cap Z' : "Z", 'decimal' : ".", 'divide' : "/", 'eight' : "8", 'equal' : "=",
        'five' : "5", 'four' : "4", 'minus' : "-", 'multiply' : "*", 'nine' : "9", 'one' : "1", 'plus' : "+", 'seven' : "7",
        'six' : "6", 'small a' : "a", 'small y' : "y", 'small z' : "z", 'three' : "3", 'two' : "2", 'zero' : "0"}

# def get_folder_name():
#     class_oder = image_dataset_from_directory(
#         "D:\\tech\\programming language\\equation solver\\dataset\\data_aug",
#         image_size=(224,224),
#         batch_size=32
#     )

#     return class_oder.class_names

folders = ['cap A', 'cap Y', 'cap Z', 'decimal', 'divide', 'eight', 'equal', 'five','four', 'minus', 'multiply', 'nine',
            'one', 'plus', 'seven','six', 'small a', 'small y', 'small z', 'three', 'two', 'zero']


def get_equation(unit):

    equation = ""
    for ch in unit:
        
        img = ch["image"]
        img = cv2.bitwise_not(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        equation += value[folders[predicted_class]]

    return equation


image_path = r"D:\tech\programming language\equation solver\static\sample\sample4.png"

with open(image_path, "rb") as image_file:
    thresh, characters = preprocess_equation(image_file)

equation = get_equation(characters)

print("Equation:", equation)
