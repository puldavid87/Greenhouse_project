import cnn_tf_models as cnn_tf 


folder_name = "Efficient"
path_model_destination = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/" + folder_name + "/"
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"


classes = 4
epochs = 20
unfreeze_layers = -20

# Define some parameters for the loader:
batch_size = 32