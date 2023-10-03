import cnn_tf_models as cnn_tf 
import data_exploration as de
import data_augmentation_keras as dak
import EfficientNetB0 as Efficient
import os
import tensorflow as tf
#Datasets
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path + "/train"
output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/augmented_images"
os.makedirs(output_path, exist_ok=True)


labels, tam_labels = de.get_labels(train_path)
dak.rotation_per_image(train_path, labels, output_path)

#Models
folder_name = "Efficient"
path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"

#variables
classes = 4
epochs = 20
unfreeze_layers = -20
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32

cnn_tf.make_folder(folder_name, path_model_destination)

train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)

efficient_model = Efficient.build_model(classes, False)
cnn_tf.unfreeze_model(efficient_model, 20) 
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
model, train_data, validation_data, test_data, callback, path_model_destination,epochs, name
cnn_tf.train_model(efficient_model, train_data, validation_data, test_data, callback, path_model_destination, 1,name="EfficientNetB0_test1")
