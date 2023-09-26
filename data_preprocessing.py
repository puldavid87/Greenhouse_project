#https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import numpy as np
from sklearn import manifold
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pandas as pd
import shutil
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

dataset_path = "C:/Users/paur/Documents/Invernadero/dataset"
destination_path = "C:/Users/paur/Documents/Invernadero/"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"


def get_labels(train_path):
    labels=[]
    tam = []
    for cont, i in enumerate((os.listdir(train_path))):
        labels.append(i)  
    for cont, i in enumerate (labels):
        num = len(os.listdir(train_path+"/"+i))
        tam.append(num)
        print("Label ", labels[cont], ": ", num)
        tam_labels = np.array(tam) 
    return labels, tam_labels 

labels,tam_labels=get_labels(train_path)

def get_flatten_dataset (train_path,labels):            
    flatten_dataset = []
    directory = []
    for i in labels:
        file_path = train_path + "/" + i
        file_names = os.listdir(file_path)
        for j in file_names:
            gray_image_path = cv2.imread(file_path + "/" + str(j))
            directory.append(str(j))
            gray_image = cv2.cvtColor(gray_image_path, cv2.COLOR_RGB2GRAY)
            if gray_image.shape[0] != 224:
                gray_image = cv2.resize(gray_image, (224,224), interpolation = cv2.INTER_AREA)  
            if gray_image.shape[1] != 224:
                gray_image = cv2.resize(gray_image, (224,224), interpolation = cv2.INTER_AREA)  
            ima = np.reshape(gray_image,50176)
                #aux.append(gray_image)
            flatten_dataset.append(ima)
    flatten_dataset = np.array(flatten_dataset)
    return flatten_dataset, directory

def tsne_method(flatten_dataset):
    flatten_dataset_in = np.array(flatten_dataset)
    tsne = manifold.TSNE(n_components=2)
    dr_dataset = tsne.fit_transform(flatten_dataset_in)
    return dr_dataset


#################################################
###################   P   C   A   ###############
#################################################
def pca_method (flatten_dataset):
    flatten_dataset_in = np.array(flatten_dataset)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flatten_dataset_in)
    return pca_result

def flatten_labels (labels, train_path):
    labels_dr = []
    j = 1
    for i in labels:
        for i in range (len(os.listdir(train_path+"/"+i))):
            labels_dr.append(j)
        j +=1    
    return labels_dr    

flatte_dataset,directory = get_flatten_dataset(train_path,labels)


pca_dataset = pca_method (flatte_dataset)

tsne_dataset = tsne_method (flatte_dataset)

y = flatten_labels(labels,train_path)

X = pd.DataFrame(pca_dataset)
X ["label"] = y
X ['directory'] = directory
print(X)
#ONE CLASS SVM



osvm = OneClassSVM(kernel='poly', nu=0.3)
aux=osvm.fit_predict(X.iloc[:,:-2])
one_svm_database=X[(aux==1)]
print(one_svm_database)

def make_new_dataset(model_name,destination_path,train_path,dataset,labels):
    folder = destination_path + "/" + str(model_name)
    train_folder = folder + "/"+"train"
    os.makedirs(folder)
    os.makedirs(train_folder)
    for i in labels:
        train_path_folder=train_folder + "/" + str(i)
        os.makedirs(train_path_folder)
        file_path = train_path + "/" + i
        file_names = os.listdir(file_path)
        for j in file_names:
            for l in dataset.iloc[:,-1]:
                if j == l:
                    shutil.copy(file_path +
                                "/" + str(j),train_path_folder 
                                + "/" + str(l))
#ISOLATION FOREST BY LABELS


iso=IsolationForest(contamination=0.3)
aux=iso.fit_predict(X.iloc[:,:-2])
isolation_database=X[(aux==1)]
print(isolation_database)

def plot_dr_figure (dr_dataset,labels_dr):
    dr_dataset = np.array(dr_dataset)
    plt.figure(figsize=(16,10))
    plt.scatter(dr_dataset [:,0],dr_dataset [:,1], c=labels_dr)
    plt.show()

print(X.iloc[:,-1])    
plot_dr_figure(X.iloc[:,:-2],X.iloc[:,-2])