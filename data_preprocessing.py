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

#################################################
###################  T  -  S N E  ###############
#################################################
def tsne_method(flatten_dataset):
    #flatten_dataset_in = np.array(flatten_dataset)
    tsne = manifold.TSNE(n_components=2)
    dr_dataset = tsne.fit_transform(flatten_dataset)
    return dr_dataset

#################################################
###################   P   C   A   ###############
#################################################
def pca_method (flatten_dataset):
    #flatten_dataset_in = np.array(flatten_dataset)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flatten_dataset)
    return pca_result

#flatten_dataset
def make_dataset_ps (dataset, y, directory):
    X = pd.DataFrame(dataset)
    X ["label"] = y
    X ['directory'] = directory
    return X
#ONE CLASS SVM

#https://scikit-learn.org/stable/modules/svm.html
# kernel_in: poly, rbf,sigmoid, linear 
def osvm_ps (X, kernel_in):
    osvm = OneClassSVM(kernel=str(kernel_in), nu=0.3)
    aux=osvm.fit_predict(X.iloc[:,:-2])
    one_svm_database=X[(aux==1)]
    print(one_svm_database)
    return one_svm_database

#0,1 >= num < = 0.5
def iso_ps (X, num):
    iso=IsolationForest(contamination=float(num))
    aux=iso.fit_predict(X.iloc[:,:-2])
    isolation_database=X[(aux==1)]
    print(isolation_database)
   
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


#plot_dr_figure(X.iloc[:,:-2],X.iloc[:,-2])
def plot_dr_figure (dr_dataset,labels_dr):
    dr_dataset = np.array(dr_dataset)
    plt.figure(figsize=(16,10))
    plt.scatter(dr_dataset [:,0],dr_dataset [:,1], c=labels_dr)
    plt.show()



def balancing_data (tam_labels, train_path, labels):
    for pos, i in enumerate(tam_labels):
        if i < max(tam_labels):
            l= max(tam_labels)-i
            file_names = os.listdir(train_path+"/"+labels[pos])
            target_images = random.sample(file_names, l)
            for m in target_images:
                shutil.copy(train_path+"/"+labels[pos] +
                                    "/" + str(m),train_path+"/"+labels[pos] 
                                    + "/"+str(l) + str(m))    
    print("Done")            