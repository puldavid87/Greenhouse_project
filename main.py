import data_exploration as de
import data_preprocessing as dp
import data_augmentation_keras as dak
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
destination_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project"

#first try
de.autentication('paulrosero/tomato-leaf-illness-detection')
de.unzipfile('tomato-leaf-illness-detection.zip')
de.check_data(dataset_path)
labels, tam_labels = de.get_labels(train_path)
de.plot_n_images(train_path,labels,3)

de.shape_labels(labels,train_path)

flatten_dataset, directory = de.get_flatten_dataset(train_path,labels)
new_labels = de.flatten_labels(labels,train_path)

tsne_dataset = dp.tsne_method(flatten_dataset)
pca_dataset = dp.pca_method(flatten_dataset)

pca_dataset = dp.make_dataset_ps(pca_dataset,new_labels,directory)

svm_dataset = dp.osvm_ps(pca_dataset, "poly")

iso_datate = dp.iso_ps(pca_dataset, 0.4)

dp.make_new_dataset ("svm",destination_path,train_path, svm_dataset,labels)


labels, tam_labels = de.get_labels(destination_path+"svm/train")

train_path=destination_path+"svm/train"

dp.balancing_data(tam_labels,train_path,labels)
               
