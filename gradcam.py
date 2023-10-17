#https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
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
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
replace2linear = ReplaceToLinear()

output_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/grad_cam"
os.makedirs(output_path, exist_ok=True)
dataset_path = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
img_size = (224,224)

image_titles = ['Dried leaves', 'Healthy leaves', 'Leaves with stains', "Leaves with yellow stains"]
img_label1 = keras.utils.load_img(test_path + "/dried_leaves/20230704_212326010_iOS.jpg", target_size=img_size)
img_label2 = keras.utils.load_img(test_path + "/healthy_leaves/20230704_210044006_iOS.jpg", target_size=img_size)
img_label3 = keras.utils.load_img(test_path + "/leaves_with_stains/20230704_220525577_iOS.jpg", target_size=img_size)
img_label4 = keras.utils.load_img(test_path + "/leaves_yellow_stains/20230704_214323288_iOS.jpg", target_size=img_size)

test_images = [img_label1,img_label2,img_label3,img_label4]

models_names = ["Efficient", "Inception", "Mobilenet", "Xception", "VGG16"]
#model_convext = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Convext/Convext_Falseunfree.h5")
model_efficient = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Falseunfree.h5")
model_inception = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Inception/Inception_Falseunfree.h5")
model_mobilenet = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Mobilenet/Mobilenet_Falseunfree.h5")
model_xception = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Xception/Xception_Falseunfree.h5")
model_vgg16 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/VGG16/vgg16_Falseunfree.h5")
models = [model_efficient, model_inception, model_mobilenet, model_xception, model_vgg16]

model_variations = ["added_top_l", "unfrozen", "augmented", "unfro_aug"]
############ Efficient  ################################
model_efficient_1 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_False.h5")
model_efficient_2 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_True.h5")
model_efficient_3 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Falseunfree.h5")
model_efficient_4 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Trueunfree.h5")

#Loading.
def visualization (image_titles):
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    for i in range (4):
        ax[i].set_title(image_titles[i], fontsize=16)
        ax[i].imshow(test_images[i])
        ax[i].axis('off')
    plt.show() 

def grad_cam_models (models, img, models_names, output_path, label, label_score):
    for i, model in enumerate (models):
        output_path_images = output_path + "/" + label + "/"
        os.makedirs(output_path_images, exist_ok=True)
        gradcam = Gradcam(model,
                    model_modifier=replace2linear,
                    clone=False) 
        score = CategoricalScore(label_score)
        cam = gradcam(score,
                np.expand_dims(np.array(img), axis=0).astype(np.float32),
                penultimate_layer=-1)
        cam = normalize (cam)
        heatmap = np.uint8(cm.jet(cam[0])[..., :4]* 255)
        plt.imshow(img)
        plt.title(models_names[i])
        plt.imshow(heatmap, cmap='jet', alpha=0.4) # overlay
        plt.savefig(output_path_images + models_names[i] + ".png")
        plt.axis('off')
        plt.show()
        
        
        
def grad_cam_labels (model, test_images, model_name, output_path):
    gradcam = Gradcam(model,
                    model_modifier=replace2linear,
                    clone=False)
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    for i, label in enumerate (image_titles):
        score = CategoricalScore(i)
        cam = gradcam(score,
                np.expand_dims(test_images[i], axis=0).astype(np.float32),
                penultimate_layer=-1)
        cam = normalize (cam)
        #y = np.argmax(model.predict(np.expand_dims(images[i], axis=0).astype(np.float32))) 
        heatmap = np.uint8(cm.jet(cam[0])[..., :4]* 255)
        ax[i].set_title(label, fontsize=16)
        ax[i].imshow(test_images[i])
        ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
        ax[i].axis('off') 
    plt.savefig(output_path + "/" + model_name + ".png")       
    plt.show() 
    

#########################################################################################
visualization (image_titles)
for i, image in enumerate (test_images):
    grad_cam_models (models, image, models_names, output_path, image_titles[i] , i)

for i, model in enumerate (models):
    grad_cam_labels (model, test_images, models_names[i],output_path)

##########################################################################################
model_variations = ["added_top_l", "unfrozen", "augmented", "unfro_aug"]
############ Efficient  ################################
model_efficient_1 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_False.h5")
model_efficient_2 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_True.h5")
model_efficient_3 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Falseunfree.h5")
model_efficient_4 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Trueunfree.h5")

efficient_models = [model_efficient_1,model_efficient_2,model_efficient_3,model_efficient_4]
output_path_effi = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/grad_cam" + "/Efficient/"
os.makedirs(output_path_effi, exist_ok=True)
for i, model in enumerate (efficient_models):
    grad_cam_labels (model, test_images, model_variations[i],output_path_effi)
    
############ Efficient  ################################
model_mobilenet_1 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Mobilenet/Mobilenet_False.h5")
model_mobilenet_2 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Mobilenet/Mobilenet_True.h5")
model_mobilenet_3 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Mobilenet/Mobilenet_Falseunfree.h5")
model_mobilenet_4 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Mobilenet/Mobilenet_Trueunfree.h5")

mobilenet_models = [model_mobilenet_1,model_mobilenet_2,model_mobilenet_3,model_mobilenet_4]
output_path_mobi = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/grad_cam" + "/Mobilenet/"
os.makedirs(output_path_mobi, exist_ok=True)
for i, model in enumerate (mobilenet_models):
    grad_cam_labels (model, test_images, model_variations[i],output_path_mobi)
    
'''
images = []
name = []
folder_name = "/leaves_yellow_stains/"
file_names = os.listdir(test_path + folder_name )
test_images = random.sample(file_names , 10)
for i in test_images:    
    img = keras.utils.load_img(test_path+folder_name+ i, target_size=img_size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = np.array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)    
    images.append(array)
    name.append(i)
    plt.imshow(mpimg.imread(test_path + folder_name+ i))
    plt.title(i)
    plt.show()    
images=np.array(images)

'''