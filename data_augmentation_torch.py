import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from torchvision.io import read_image
import numpy as np
from torchvision.utils import save_image
# Define the path to your image
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path + "/train"
output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/augmented_images_torch"
os.makedirs(output_path, exist_ok=True)

labels = [
    'dried_leaves',
    'healthy_leaves',
    'leaves_with_stains',
    'leaves_yellow_stains']


def rotation_per_image_torch(train_path, labels, output_path):
    for i in labels:
        os.makedirs(output_path + "/" + i, exist_ok=True)
    for i in labels:
        file_names = os.listdir(train_path + "/" + i)
        img = random.sample(file_names, 1)
        # Load the image and apply data augmentation
        image = Image.open(train_path + "/" + i + "/" + img[0])
        # Create 20 augmented versions of the image
        augmented_images = [transform(image) for _ in range(20)]
        cont = 0
        for augmented_image in augmented_images:
            augmented_image = transforms.ToPILImage()(augmented_image)
            save_path = os.path.join(
                output_path + "/" + i,
                f"augmented_image_{i}_{cont}.jpg")
            augmented_image.save(save_path)
            cont += 1
    return augmented_images


# Define a transform that includes data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),
    # Randomly rotate the image by up to 30 degrees
    transforms.RandomRotation(40),
    # Adjust brightness, contrast, saturation, and hue
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Randomly crop and resize the image to 224x224 pixels
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])


def plot_data_augmentation(image, augmented_images):
    # Display the original and augmented images
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 5, 1)
    plt.title("Original Image")
    plt.imshow(image)
    # plt.axis('off')
    #plt.subplot(2, 1, 2)
    for i, augmented_image in enumerate(augmented_images):
        plt.subplot(4, 5, i + 2)
        plt.title(f"Augmented Image {i + 1}")
        # Convert tensor to (H, W, C) for display
        plt.imshow(augmented_image.permute(1, 2, 0))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


rotation_per_image_torch(train_path, labels, output_path)
