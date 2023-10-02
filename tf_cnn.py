import cnn_tf_models as cnn_tf 
import data_exploration as de
import data_augmentation_keras as dak
import os

#Datasets
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path + "/train"
output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/augmented_images"
os.makedirs(output_path, exist_ok=True)


#Models
folder_name = "Efficient"
path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"

#variables
classes = 4
epochs = 20
unfreeze_layers = -20
# Define some parameters for the loader:
batch_size = 32

labels = de.get_labels(train_path)

rotation_per_image(train_path, labels, output_path)