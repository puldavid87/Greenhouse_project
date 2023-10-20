#https://iq.opengenus.org/inception-v3-model-architecture/

import tensorflow as tf
import cnn_tf_models as cnn_tf 
# import necessary layers  
from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
#Datasets
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
test_dir = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset/test"
model_vgg16 = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/VGG16/vgg16_Falseunfree.h5")
#variables
classes = 4
epochs = 20
unfreeze_layers = -20
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)
#####################################################################

# input
def build_model(num_classes):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    # 1st Conv Block
    model.add(tf.keras.layers.Conv2D(8,3,activation="relu"))
    model.add(tf.keras.layers.Conv2D(8, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2, padding="valid"))
    # 2nd Conv Block
    model.add(tf.keras.layers.Conv2D(16, 3, activation="relu"))
    model.add(tf.keras.layers.Conv2D(16, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2, padding="valid"))
    # 3rd Conv block
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2, padding="valid"))
    # 4th Conv block
    x = Conv2D (filters =64, kernel_size =3, padding ='valid', activation='relu')(x)
    x = Conv2D (filters =64, kernel_size =3, padding ='valid', activation='relu')(x)
    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(units = 16, activation ='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=["accuracy"])
    return model

s_model_1 = build_model(4)



################################################

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.5,
        temperature=2,
    ):
        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    def train_step(self, data):
        # Unpack data
        x, y = data
        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        #model = ...  # create the original model
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            # Forward pass of student
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss +  distillation_loss
        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_prediction = self.student(x, training=False)
        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    

# Initialize and compile distiller
distiller = Distiller(student=s_model_1, teacher=model_vgg16)
distiller.compile(
    optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=['acc'],
    student_loss_fn=CategoricalCrossentropy(from_logits=True),
distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=2)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)
# Distill teacher to student
distiller_hist = distiller.fit(train_data, validation_data = validation_data, epochs=10, validation_steps=len(validation_data),
              steps_per_epoch = len(train_data), callbacks=[callback] )

s_model_1.save("C:/Users/paur/Documents/Invernadero/Models/VGG16/destiller.h5")

s_model_1.evaluate (test_data, verbose = 1)
import matplotlib.pyplot as plt 
plt.figure(1)  
# summarize history for accuracy  
plt.subplot(211)  
plt.plot(distiller_hist.history['acc'])  
plt.plot(distiller_hist.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid'], loc='lower right')  
 # summarize history for loss  
plt.subplot(212)  
plt.plot(distiller_hist.history['student_loss'])  
plt.plot(distiller_hist.history['val_student_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid'], loc='upper right')  
plt.show()
plt.tight_layout()
backend = None
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = tf.keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(axis=3, scale=False)(x)
    x = tf.keras.layers.Activation('relu', name=name)(x)
    return x

'''
def small_inception (num_clases):
    channel_axis = 3
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1. / 127.5, offset=-1.)(inputs)
    x = conv2d_bn(inputs_re, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
     # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')
    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis = channel_axis,
        name = 'mixed1')
    
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis = channel_axis,
        name ='mixed3')
    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis = channel_axis,
        name='mixed8')
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, x, name='inception_v3')
    
    return model
'''
def small_inception (num_clases):
    channel_axis = 3
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    inputs_re = tf.keras.layers.experimental.preprocessing.Rescaling(
        scale=1. / 127.5, offset=-1.)(inputs)
    x = conv2d_bn(inputs_re, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 16, 3, 3, padding='valid')
    x = conv2d_bn(x, 32, 3, 3)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 40, 1, 1, padding='valid')
    x = conv2d_bn(x, 96, 3, 3, padding='valid')
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
     # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 32, 1, 1)
    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5)
    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')
    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 32, 1, 1)
    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5)
    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch_pool = tf.keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = tf.keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis = channel_axis,
        name = 'mixed1')
    
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 192, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 48, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis = channel_axis,
        name ='mixed3')
    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 96, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 160, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 96, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 96, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 96, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 96, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis = channel_axis,
        name='mixed8')
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs, x, name='inception_v3')
    
    return model    
inception_s =  small_inception (4)

model_inception = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Inception/Inception_Falseunfree.h5")

# Initialize and compile distiller
distiller = Distiller(student= inception_s, teacher=model_inception)
distiller.compile(
    optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=['acc'],
    student_loss_fn=CategoricalCrossentropy(from_logits=True),
distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=4)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)
# Distill teacher to student
distiller_hist = distiller.fit(train_data, validation_data = validation_data, epochs=1, validation_steps=len(validation_data),
              steps_per_epoch = len(train_data), callbacks=[callback] )

inception_s.save("C:/Users/paur/Documents/Invernadero/Models/Inception/Inception_S.h5")

inception_s.evaluate(test_data, verbose = 1)