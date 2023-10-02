import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import numpy as np

folder_name = "Inception"
path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
classes = 4
epochs = 20
unfreeze_layers = -20

# Define some parameters for the loader:
batch_size = 32

#make_folder(folder_name, path_model_destination)
def make_folder(folder_name, path_destination):
    """Creates a folder with the given name at the destination path
    
    Args:
        folder_name: Name of the folder to create
        path_destination: Path where the folder will be created
    Returns: 
        None: Does not return anything
    
    - Creates the folder with the given name at the destination path using os.makedirs()
    - Sets exist_ok flag to True to avoid error if folder already exists
    - Prints a message to confirm folder creation
    """
    os.makedirs(path_destination, exist_ok=True)
    print("Folder ", folder_name, "was created")

#train_data, validation_data, test_data = split_tratin_test_set()
def split_tratin_test_set():
    """
    Splits training and test datasets
    Args: 
        train_dir: Path to training directory
        test_dir: Path to test directory
    Returns:
        train_data: Training dataset
        validation_data: Validation dataset 
        test_data: Test dataset
    Splits training images into train and validation sets. Imports test images separately. Converts all images to specified height and width.
    - Imports data from train and test directories 
    - Splits train data into train and validation sets
    - Converts images to specified height and width
    - Returns train, validation and test datasets
    """
    train_dir = path_data_source + "/" + "train"
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


def unfreeze_model(model, num):
    """Unfreeze top layers of a model
    Args:
        model: The model to unfreeze layers for
        num: The number of top layers to unfreeze
    Returns: 
        model: The model with top layers unfrozen
    Processing Logic:
        - Loop through layers from num to the end of the model
        - Set trainable to True for layers that are not BatchNormaliztion layers
        - Compile the model with an Adam optimizer with learning rate of 1e-4"""
    # We unfreeze the top num layers while leaving BatchNorm layers frozen
    for layer in model.layers[num:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )


def results(model, test_data, name):
    """
    Calculates model performance metrics on test data
    Args:
        model: The trained model
        test_data: The test data
        name: The name of the model
    Returns: 
        None: Does not return anything, just prints metrics
    Processing Logic:
        - Predicts classes for test data using model
        - Extracts true and predicted classes
        - Calculates precision, recall, f1 score and mean absolute error
        - Prints classification report and confusion matrix
        - Plots and saves confusion matrix
    """
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
        path_model_destination +
        "cf_efficient_" +
        str(name) +
        ".png")
    # Display the visualization of the Confusion Matrix.
    plt.show()


def plot_loss_curves(history, name):
    """
    Plots loss and accuracy curves from the Keras model history
    Args:
        history: Keras model history object
        name: Name of the model
    Returns: 
        None: Does not return anything, saves figures to file
    Processing Logic:
        - Extracts loss and accuracy values from history
        - Plots loss vs epochs and saves figure
        - Plots accuracy vs epochs and saves figure
    """
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
    plt.savefig(path_model_destination +
                "Loss_" +
                str(name) + ".png")

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(path_model_destination +
                "ACC_" +
                str(name) + ".png")


#model, history = first_model(classes)
def first_model(classes, name="EfficientNetB0_test1"):
    model = build_model(num_classes=classes, aprov_pre=False)
    start = datetime.now()
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
    end = datetime.now()
    # find difference loop start and end time and display
    td = (end - start)
    # model.save(
    #            path_model_destination +
    #            name+
    #                ".h5")
    print("Exceuction time:", td)
    results(model, test_data, name)
    plot_loss_curves(history, name)
    return model, history

#Loading...
def compare_historys(
        original_history,
        new_history,
        initial_epochs=5, name="EfficientNetB0"):
    """
    Compares two model history objects.
    Args:
        original_history: {The original model history object in one line}
        new_history: {The new model history object to combine with the original in one line}
        initial_epochs: {The number of epochs in the original training in one line} 
        name: {The name of the model in one line}
    Returns: 
        None: {Does not return anything, just plots the combined training history in one line}
    Processing Logic:
        - Get measurements from original history
        - Combine measurements from original and new histories
        - Make plots comparing training and validation accuracy and loss over total epochs
    """
        
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
             label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(),
             label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig(path_model_destination +
                "FINE_" +
                str(name) + ".png")











#model_1, history_1 = second_model(classes)
def second_model(classes, name="EfficientNetB0_test2"):
    """
    Builds and trains a neural network model on provided data.
    
    Args:
        classes: Number of output classes
        name: Name of the model
    Returns: 
        model: Trained Keras model
        history: Training history object
    
    Processing Logic:
    - Builds a EfficientNetB0 model with provided number of classes
    - Fits the model on training data for 1 epoch 
    - Validates on validation data
    - Records training time
    - Evaluates model on test data and plots loss curves
    """
    model = build_model(num_classes=classes, aprov_pre=False)
    unfreeze_model(model, unfreeze_layers)
    start = datetime.now()
    history = model.fit(train_data,
                        epochs=1,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        # Go through less of the validation data so epochs
                        # are faster (we want faster experiments!)
                        validation_steps=int(len(validation_data)),
                        callbacks=[callback],
                        #
                        verbose=1)
    end = datetime.now()
    # find difference loop start and end time and display
    td = (end - start)
    print("Exceuction time:", td)
    # model.save(
    #            path_model_destination +
    #            name+
    #                ".h5")
    results(model, test_data, name)
    plot_loss_curves(history, name)
    return model, history


#model_2, history_2 = third_model(classes)
def third_model(classes, name = "EfficientNetB0_test3"):
    """
    Builds and trains a neural network model.
    
    Args:
        classes: Number of output classes
        name: Name of the model
    Returns: 
        model: Trained Keras model
        history: Training history object
    
    Processing Logic:
    - Builds a EfficientNetB0 model with the given number of classes
    - Fits the model on the training data for 1 epoch
    - Validates on validation data 
    - Prints the execution time
    - Evaluates the model on test data and plots loss curves
    """
    model = build_model(num_classes = classes, aprov_pre=True)
    unfreeze_model(model, unfreeze_layers)
    start = datetime.now()
    history = model.fit(train_data,
                        epochs=1,
                        steps_per_epoch=len(train_data),
                        validation_data=validation_data,
                        # Go through less of the validation data so epochs
                        # are faster (we want faster experiments!)
                        validation_steps=int(len(validation_data)),
                        callbacks=[callback],
                        #
                        verbose=1)
    end = datetime.now()
    # find difference loop start and end time and display
    td = (end - start)
    print("Exceuction time:", td)
    # model.save(
    #            path_model_destination +
    #            name+
    #                ".h5")
    results(model, test_data, name)
    plot_loss_curves(history, name)
    return model, history


#compare_historys(history, history_1, initial_epochs = 5, name = "EfficientNetB0")
