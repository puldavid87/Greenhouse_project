import data_exploration as de
import data_preprocessing as dp
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
destination_path = "C:/Users/paur/Documents/Invernadero/"

#first try
de.autentication('paulrosero/tomato-leaf-illness-detection')
de.unzipfile('tomato-leaf-illness-detection.zip')
de.check_data(dataset_path)
labels, tam_labels = de.get_labels(train_path)
de.plot_n_images(train_path,labels,3)

de.shape_labels(labels,train_path)

flatten_dataset = de.get_flatten_dataset(train_path,labels)

new_labels = de.flatten_labels(labels,train_path)

tsne_dataset = dp.tsne_method(flatten_dataset)

