import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
# Display
import cv2
from IPython.display import Image, display
import matplotlib.cm as cm
from tensorflow.keras.applications.inception_v3 import preprocess_input as input_inception
from tensorflow.keras.applications.efficientnet import preprocess_input as input_efficient
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

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
    
    images.append(array)
    plt.imshow(mpimg.imread(test_path+"/"+i + "/"+ str(image[0])))
    plt.title(i)
    plt.show()
    
images=np.array(images)


model = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Inception/Inception_Falseunfree.h5")
gradcam = Gradcam (model)

replace2linear = ReplaceToLinear()

gradcam = Gradcam(model,
                  model_modifier=replace2linear,
                  clone=False)

image_titles = ['Dried leaves', 'Healthy leaves', 'Leaves with stains', "Leaves with yellow stains"]

f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
for i, label in enumerate (image_titles):
    score = CategoricalScore(i)
    cam = gradcam(score,
              np.expand_dims(images[i], axis=0).astype(np.float32),
              penultimate_layer=-1)
    cam = normalize (cam)
    #y = np.argmax(model.predict(np.expand_dims(images[i], axis=0).astype(np.float32))) 
    heatmap = np.uint8(cm.jet(cam[0])[..., :4]* 255)
    ax[i].set_title(label, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
    ax[i].axis('off')
plt.show() 


