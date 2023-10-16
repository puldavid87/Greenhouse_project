import cnn_tf_models as cnn_tf 
import data_exploration as de
import data_augmentation_keras as dak
import EfficientNetB0 as Efficient
import InceptionV3 as Inception
import MobileNetV2 as Mobilenet
import vgg16 as VGG16
import Xception 
import ConvNext 
import os
import mlflow
import tensorflow as tf

#Datasets
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path + "/train"
output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/augmented_images"
os.makedirs(output_path, exist_ok=True)


labels, tam_labels = de.get_labels(train_path)
dak.rotation_per_image(train_path, labels, output_path)

#variables
classes = 4
epochs = 20
unfreeze_layers = -20
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

models = ["Efficient", "Inception", "Mobilenet", "Xception", "Convext", "VGG16"]
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"
train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)

data_augmentation = [False, True]

for augmentation in data_augmentation:
    efficient_model = Efficient.build_model(classes, augmentation)
    inceptionV3 = Inception.build_model(classes, augmentation)
    mobilenet_model = Mobilenet.build_model(classes, augmentation)
    xception_model = Xception.build_model(classes, augmentation)
    convnext_model =  ConvNext.build_model(classes, augmentation) 
    vgg16_model = VGG16.build_model(classes, augmentation)
    #Models
    ml_models = [efficient_model,inceptionV3,mobilenet_model,xception_model,convnext_model, vgg16_model]
    for i, folder_name in enumerate (models):
        mlflow.set_experiment(models[i]) 
        with mlflow.start_run(): 
            with mlflow.start_run(nested=True):
                path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
                cnn_tf.make_folder(folder_name, path_model_destination) 
                model, history = cnn_tf.train_model(ml_models[i], train_data, validation_data, test_data, 
                            callback, path_model_destination,epochs,name=str(folder_name) 
                            + "_" + str(augmentation))
                mlflow.sklearn.log_model(model, str(folder_name) 
                            + "_" + str(augmentation))
            with mlflow.start_run(nested=True):   
                cnn_tf.unfreeze_model(ml_models[i], unfreeze_layers )
                model, history  =cnn_tf.train_model(ml_models[i], train_data, validation_data, test_data, 
                            callback, path_model_destination,epochs,name=str(folder_name) 
                            + "_" + str(augmentation) + "unfree")
                mlflow.sklearn.log_model(model, str(folder_name) 
                            + "_" + str(augmentation) + "unfree")

        
    