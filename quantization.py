# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:41:41 2023

@author: paur
"""
from sklearn import metrics
import tensorflow as tf
import os
import numpy as np
import pathlib
import cnn_tf_models as cnn_tf 
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
model_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/"
train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)

def representative_data_gen():
  for input_value,_ in train_data.take(100):
    input_value=np.expand_dims(input_value[0], axis=0).astype(np.float32)
    yield [input_value]
    

#Loading...
def to_tf_lite (model, model_path, name):   
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(model_path)
    full_name= str(name) + ".tflite"
    tflite_model_file = tflite_models_dir/full_name
    tflite_model_file.write_bytes(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    evaluate_model(interpreter, test_data, name,model_path)
#Convert using integer-only quantization

def to_int8 (model, model_path, name): 
    tflite_models_dir = pathlib.Path(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    full_name= str(name) + "int8.tflite"
    tflite_model_quant_int = converter.convert()
    tflite_model_quant_int_file = tflite_models_dir/full_name
    tflite_model_quant_int_file.write_bytes(tflite_model_quant_int)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_int_file))
    evaluate_model(interpreter, test_data, name,model_path)
        
def evaluate_model(interpreter, test_data, name,model_path):
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    cont=0
    y_test = []    
    y_pred = []
    # Run predictions on every image in the "test" dataset.
    input_details = interpreter.get_input_details()[0]
    for test_image,test_label in test_data:
        for image in test_image:
            y_test.append(np.argmax(test_label[cont,:]))
            if input_details['dtype'] == np.uint8:
                image_to_tensor = np.expand_dims(image, axis=0).astype(np.uint8)
            else:
                image_to_tensor = np.expand_dims(image, axis=0).astype(np.float32)
            cont+=1
            interpreter.set_tensor(input_index, image_to_tensor)
            # Check if the input type is quantized, then rescale input data to uint8
            # Run inference.
            interpreter.invoke()
            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            if input_details['dtype'] == np.uint8:
                y_pred.append(np.argmax((output()[0])/255))
            else:
                y_pred.append(np.argmax((output()[0])))
        cont = 0        
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
        model_path +
        "confusion_matrix" +
        str(name) +
        ".png")
    # Display the visualization of the Confusion Matrix.
    plt.show()

models = ["Efficient", "Inception", "Mobilenet", "Xception", "Convext"]
data_augmentation = ["False", "True"]
word = ["unfree"]

for i in models :
    for j in data_augmentation:        
        model = tf.keras.models.load_model(model_path + i + "/" + i + "_" + j + ".h5"))
        to_tf_lite(model, model_path, i + "_" + j + "_lite")
        to_int8(model, model_path, i + "_" + j + "int8_lite")
        for k in word:
            model = tf.keras.models.load_model(model_path + i + "/" + i + "_" + j + k +".h5")
            to_tf_lite(model, model_path, i + "_" + j + k + "_lite")
            to_int8(model, model_path, i + "_" + j + k + "int8_lite")
            

