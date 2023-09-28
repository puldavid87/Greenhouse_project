# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:05:12 2023

@author: paur
"""
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tensorflow.keras.applications import EfficientNetB0

import numpy as np

path_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/TF_MODELS"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"

classes = 4
epochs = 20
unfreeze_layers = [-20]

# Define some parameters for the loader:
batch_size = 32
img_height = 224
img_width = 224

def split_tratin_test_set():
    train_dir = path_data_source  + "/" + "train"
    # Import data from directories and turn it into batches
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     seed=123,
                                                                     label_mode="categorical",
                                                                     batch_size=batch_size,  # number of images to process at a time
                                                                     validation_split=0.2,
                                                                     subset="training",
                                                                     image_size=(img_height, img_width))  # convert all images to be 224 x 224

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                          seed=123,
                                                                          label_mode="categorical",
                                                                          batch_size=batch_size,  # number of images to process at a time
                                                                          validation_split=0.2,
                                                                          subset="validation",
                                                                          image_size=(img_height, img_width))  # convert all images to be 224 x 224

    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                    seed=123,
                                                                    label_mode="categorical",
                                                                    batch_size=batch_size,  # number of images to process at a time
                                                                    image_size=(img_height, img_width))  # convert all images to be 224 x 224
    return train_data, validation_data, test_data

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


def unfreeze_model(model, num):
    # We unfreeze the top num layers while leaving BatchNorm layers frozen
    for layer in model.layers[num:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )


def results(model, test_data, name):
    y_pred = []
    results = model.predict(test_data)
    for i in results:
        y_pred.append(np.argmax(i))

    y_test = []
    for test_image, test_label in test_data:
        for t in test_label:
            print(np.array(t))
            y_test.append(np.argmax(t))
    print("")
    print(
        "Precision: {}%".format(
            100 *
            metrics.precision_score(
                y_test,
                y_pred,
                average="weighted")))
    print(
        "Recall: {}%".format(
            100 *
            metrics.recall_score(
                y_test,
                y_pred,
                average="weighted")))
    print(
        "f1_score: {}%".format(
            100 *
            metrics.f1_score(
                y_test,
                y_pred,
                average="weighted")))
    print("Error: {}%".format(metrics.mean_absolute_error(y_test, y_pred)))
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print('\\Report\n')
    print(report)

    ax = plt.plot()
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.savefig(
        path_destination +
        "cf_efficient_" +
        str(name) +
        ".png")
    # Display the visualization of the Confusion Matrix.
    plt.show()


def plot_loss_curves(history, name):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig( path_destination +
                    "Loss_"+
                str(name)+".png")

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig( path_destination +
                "ACC_"+
                str(name)+".png")

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

train_data, validation_data, test_data = split_tratin_test_set()
            
def first_model (classes, pre=False, name = "EfficientNetB0_test1"):            
    model = build_model(num_classes=classes, aprov_pre=pre)
    history = model.fit(train_data,
                            epochs=1,
                            steps_per_epoch=len(train_data),
                            validation_data=validation_data,
                            # Go through less of the validation data so epochs are
                            # faster (we want faster experiments!)
                            validation_steps=int(len(validation_data)),
                            callbacks=[callback],
                            verbose=1,
                            )
    model.save(
                path_destination +
                name+
                    ".h5")
    
    plot_loss_curves(history, name)
    results (model, test_data, name)
    return model, history

model, history = first_model(classes) 


models = []
for j in datasets:
    train_data, test_data = split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=classes, aprov_pre=False)
            unfreeze_model(model, l)
            start = datetime.now()
            hist = model.fit(train_data,
                             epochs=1,
                             steps_per_epoch=len(train_data),
                             validation_data=test_data,
                             # Go through less of the validation data so epochs
                             # are faster (we want faster experiments!)
                             validation_steps=int(len(test_data)),
                             # 
                             verbose=1)
            end = datetime.now()
            # end_cpu=psutil.cpu_percent(interval=None)
            # find difference loop start and end time and display
            td = (end - start)
            # td_cpu=(end_cpu-start_cpu)
            print("----------------- MODEL+FINE T----------------------")
            cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
            avg = sum(cpu_percent_cores) / len(cpu_percent_cores)
            cpu_percent_total_str = ('%.2f' % avg)
            cpu_percent_cores_str = [
                ('%.2f' % x) + '%' for x in cpu_percent_cores]
            df1 = [{'dataset': j,
                    'epochs': i,
                    'DA': 0,
                    'layers': l,
                    'train': round(hist.history['accuracy'][-1],
                                   4),
                    'test':round(hist.history['val_accuracy'][-1],
                                 4),
                    'exec.time':td,
                    'ram':psutil.virtual_memory()[3] / 1000000000,
                    'cpu': cpu_percent_total_str}]
            df1 = pd.DataFrame(data=df1)
            df = pd.concat([df, df1])
            print('Total: {}'.format(cpu_percent_total_str))
            print(
                'Individual CPUs: {}'.format(
                    '  '.join(cpu_percent_cores_str)))
            #print("CPU utilization: ", td_cpu)
            print("Dataset: ", j, "epochs: ", i, "layers: ", l)
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_precent()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times())
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print("---------------------------------------------------")
            models.append(hist)
            #results (model,test_data,j, "test2")
            #model.save(path_destination+"efficient" + "_test2_" + str(j))
            model.save(
                path_destination +
                "efficient_s" +
                "_test2_" +
                str(j) +
                ".h5")


training_results_fine(datasets, "test2")
models = []

for j in datasets:
    train_data, test_data = split_tratin_test_set(j)
    for i in epochs_vector:
        for l in unfreeze_layers:
            model = build_model(num_classes=classes, aprov_pre=True)
            unfreeze_model(model, l)
            start = datetime.now()
            # start_cpu=psutil.cpu_percent(interval=1)
            hist = model.fit(train_data,
                             epochs=i + 5,
                             steps_per_epoch=len(train_data),
                             validation_data=test_data,
                             # Go through less of the validation data so epochs
                             # are faster (we want faster experiments!)
                             validation_steps=int(len(test_data)),
                             callbacks=[callback],
                             verbose=1)
            end = datetime.now()
            td = (end - start)
            # td_cpu=(end_cpu-start_cpu)
            print("----------------- MODEL+PRE+FINE T-----------------")
            cpu_percent_cores = psutil.cpu_percent(interval=2, percpu=True)
            avg = sum(cpu_percent_cores) / len(cpu_percent_cores)
            cpu_percent_total_str = ('%.2f' % avg)
            cpu_percent_cores_str = [
                ('%.2f' % x) + '%' for x in cpu_percent_cores]
            df1 = [{'dataset': j,
                    'epochs': i,
                    'DA': 1,
                    'layers': l,
                    'train': round(hist.history['accuracy'][-1],
                                   4),
                    'test':round(hist.history['val_accuracy'][-1],
                                 4),
                    'exec.time':td,
                    'ram':psutil.virtual_memory()[3] / 1000000000,
                    'cpu': cpu_percent_total_str}]
            df1 = pd.DataFrame(data=df1)
            df = pd.concat([df, df1])
            print('Total: {}'.format(cpu_percent_total_str))
            print(
                'Individual CPUs: {}'.format(
                    '  '.join(cpu_percent_cores_str)))
            print("Dataset: ", j, "epochs: ", i, "layers: ", l)
            #print("CPU utilization: ", td_cpu)
            print(f"The time of execution of above program is : {td}ms")
            # Calling psutil.cpu_times()for 4 seconds
            print('The CPU usage is: ', psutil.cpu_times())
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # Getting usage of virtual_memory in GB ( 4th field)
            print("---------------------------------------------------")
            models.append(hist)
            #results (model,test_data,j, "test3")
            #model.save(path_destination+"efficient" + "_test3_" + str(j))
            model.save(
                path_destination +
                "efficient_s" +
                "_test3_" +
                str(j) +
                ".h5")

training_results_fine(datasets, "test3")


df.to_csv(path_destination + "efficient_s.csv", index=False)
