import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.cm as cm

dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
img_size = (224,224)
images = []
for i in os.listdir(test_path):
    file_names = os.listdir(test_path + "/" + i + "/")
    image = random.sample(file_names , 1)
    img = keras.utils.load_img(test_path+"/"+i + "/"+ str(image[0]), target_size=img_size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = np.array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    #array = np.expand_dims(array/255, axis=0)
    images.append(array)
    plt.imshow(mpimg.imread(test_path+"/"+i + "/"+ str(image[0])))
    plt.title(i)
    plt.show()
    
images=np.array(images)

from tensorflow.keras.applications.inception_v3 import preprocess_input 

X = preprocess_input (images)

from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([0,1,2,3])

# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return (output[0][1], output[1][1], output[2][413])
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear
    return cloned_model

from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
model = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Greenhouse_project/Models/Efficient/Inception_test1.h5")
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

gradcam = Gradcam (model,
                   model_modifier=replace2linear,
                   clone=True)

model.summary()
#(None, 4) 
cam = gradcam(score,
              X,
              penultimate_layer=-1)

cam = normalize(cam)
image_titles = ['Goldfish', 'Bear', 'Assault rifle', "pesho"]

f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :4] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.3) # overlay
    ax[i].axis('off')
plt.tight_layout()
plt.show()


from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
gradcam = GradcamPlusPlus(model,
                          model_modifier=replace2linear,
                          clone=True)

# Generate heatmap with GradCAM++
cam = gradcam(score,
              X,
              penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()


from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus

# Create ScoreCAM object
scorecam = Scorecam(model)

# Generate heatmap with ScoreCAM
cam = scorecam(score, X, penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()