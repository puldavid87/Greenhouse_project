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


img_height = 299
img_width = 299



#Loading...
def build_model(num_classes, aprov_pre):
    """
    Builds a transfer learning model
    Args:
        num_classes: {Number of classes for the final output layer}
        aprov_pre: {Whether to use preprocessing augmentation}
    Returns: 
        model: {The compiled Keras model}
    Processing Logic:
        - Builds the InceptionV3 model as feature extractor
        - Optionally adds preprocessing augmentation
        - Flattens and adds final Dense layer 
        - Compiles model with categorical crossentropy loss
    """
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
