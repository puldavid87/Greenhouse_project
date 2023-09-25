import data_exploration as de

dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
#first try
de.autentication('paulrosero/tomato-leaf-illness-detection')
de.unzipfile('tomato-leaf-illness-detection.zip')
de.check_data(dataset_path)
labels, tam_labels = de.get_labels(train_path)
de.plot_n_images(train_path,labels,3)

de.shape_labels(labels,train_path)

flatten_dataset = de.get_flatten_dataset(train_path,labels)

tsne_data = de.tsne_method(flatten_dataset)

pca_data = de.pca_method(flatten_dataset)

new_labels = de.flatten_labels(labels,train_path)

de.plot_dr_figure(tsne_data,new_labels )
