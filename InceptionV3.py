import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tensorflow.keras.applications import InceptionV3

import numpy as np

folder_name = "Inception"
path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"

classes = 4
epochs = 20
unfreeze_layers = -20

# Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224

img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.40),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomContrast(
            factor=0.2),
    ],
    name="img_augmentation",
)


def build_model(num_classes, aprov_pre):
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1. / 127.5, offset=-1.)(inputs)
    if aprov_pre:
        y = img_augmentation(inputs_re)
        model_input = InceptionV3(
            include_top=False,
            input_tensor=y,
            weights="imagenet")
        print("preprocessing:", aprov_pre)
    else:
        model_input = InceptionV3(
            include_top=False,
            input_tensor=inputs_re,
            weights="imagenet")
        print("preprocessing:", aprov_pre)
    # Freeze the pretrained weights
    model_input.trainable = False
    x = model_input.output
    x = tf.keras.layers.Flatten()(x)
    predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, predictions)
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model
