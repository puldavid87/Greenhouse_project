# -*- coding: utf-8 -*-

import zipfile
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
from kaggle.api.kaggle_api_extended import KaggleApi


dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"

# link: 'paulrosero/tomato-leaf-illness-detection'
def autentication(link):
    str(link)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(link)

#file_name : 'tomato-leaf-illness-detection.zip'
def unzipfile(file_name):
    str(file_name)
    if os.path.isfile(file_name)==True:
        with zipfile.ZipFile(file_name,'r') as zipref:
            zipref.extractall()
    else:
        print("File not found")
        
def check_data(dataset):
    for dirpath, dirnames, filenames in os.walk(dataset):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#train_path
#get  labels and tam
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
        
def view_n_images(target_dir, target_class,n):
    target_path = target_dir+"/"+target_class
    file_names = os.listdir(target_path)
    target_images = random.sample(file_names, n)   
    # Plot images
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(target_images):
        img_path = target_path + "/" + img
        plt.subplot(1, 3, i+1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(target_class)
        plt.axis("off")

def plot_n_images(train_path,labels,n):
    for i in labels:
     view_n_images(train_path,i,n)
        
def shape_labels(labels,train_path):

    for i in labels:
        target_path = train_path+"/"+i
        file_names = os.listdir(target_path)
        target_image = random.sample(file_names, 1)
        img = mpimg.imread(target_path + "/" + str(target_image[0]))
        print(i, ": ", img.shape)
        print(img.shape[0]*img.shape[1])


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

def flatten_labels (labels, train_path):
    labels_dr = []
    j = 1
    for i in labels:
        for i in range (len(os.listdir(train_path+"/"+i))):
            labels_dr.append(j)
        j +=1    
    return labels_dr                