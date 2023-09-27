import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import random
from PIL import Image
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path + "/train"
output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/augmented_images"
os.makedirs(output_path, exist_ok=True)
labels = [
    'dried_leaves',
    'healthy_leaves',
    'leaves_with_stains',
    'leaves_yellow_stains']

def rotation_per_image (train_path,labels, output_path):
    for i in labels:
        os.makedirs(output_path + "/" + i, exist_ok=True)
    for i in labels:    
        file_names = os.listdir(train_path + "/" + i)
        img = random.sample(file_names, 1)
        image = load_img(train_path + "/" + i + "/" + str(img[0]))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        cont = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented_image = array_to_img(batch[0])
            save_path = os.path.join(output_path + "/" + i, f"augmented_image_{i}_{cont}.jpg")
            augmented_image.save(save_path)
            cont += 1
            
            if cont >= 20:  # Generate 20 augmented images (you can change this number)
                break
        plot_data_augmentation(image, output_path+"/"+i)  
                  
datagen = ImageDataGenerator(
    rotation_range=40,  # Rotate images randomly up to 40 degrees
    zoom_range=0.1,  # Zoom in or out by up to 20%
    width_shift_range=0.1, 
    height_shift_range=0.1,
    brightness_range=[0.4,1.5],
    horizontal_flip=True,  # Flip the image horizontally
    vertical_flip=True,
    #fill_mode='nearest'  # Fill pixels with the nearest available value
)


def plot_data_augmentation (image, output_path):
    # Display the original and augmented images
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(2, 1, 2)
    augmented_images = os.listdir(output_path)
    for l, augmented_image_name in enumerate(augmented_images):
        augmented_image_path = os.path.join(output_path, augmented_image_name)
        augmented_image = load_img(augmented_image_path)
        plt.subplot(4, 5, l + 1)
        plt.imshow(augmented_image)
        plt.title(f"Augmented Image {l + 1}")

    plt.tight_layout()
    plt.show()

rotation_per_image (train_path,labels, output_path)
