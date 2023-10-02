# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:05:12 2023

@author: paur
"""
import os
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tensorflow.keras.applications import EfficientNetB0

import numpy as np

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
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
    """
    Builds a transfer learning model for image classification
    Args:
        num_classes: {Number of classes for the classification task} 
        aprov_pre: {Whether to use preprocessing augmentation}
    Returns: 
        model: {The Keras model}
    Processing Logic:
        - Builds an EfficientNetB0 model with pretrained weights
        - Freezes the pretrained weights
        - Adds global average pooling, dropout and prediction layers on top
        - Compiles the model with Adam optimizer and categorical crossentropy loss
    """
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
        
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    if aprov_pre:
        x = img_augmentation(inputs)
        model = EfficientNetB0(
            include_top=False,
            input_tensor=x,
            weights="imagenet")
        print("preprocessing:", aprov_pre)
    else:
        model = EfficientNetB0(
            include_top=False,
            input_tensor=inputs,
            weights="imagenet")
        print("preprocessing:", aprov_pre)
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model





