from utils import *
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D
def EfficientNet_B0(channels,
                    expansion_coefs,
                    repeats,
                    strides,
                    kernel_sizes,
                    d_coef,
                    w_coef,
                    r_coef,
                    dropout_rate,
                    se_ratio = 0.25,
                    classes=4):
   
    inputs = Input(shape=(224, 224, 3))
    
    stage1 = ConvBlock(inputs,
                       filters=32,
                       kernel_size=3,
                       stride=2)
    
    stage2 = MBConvBlock(stage1, 
                         scaled_channels(channels[0], w_coef),
                         scaled_channels(channels[1], w_coef),
                         kernel_sizes[0],
                         expansion_coefs[0],
                         se_ratio,
                         strides[0],
                         scaled_repeats(repeats[0], d_coef),
                         dropout_rate=dropout_rate)
    
    stage3 = MBConvBlock(stage2, 
                         scaled_channels(channels[1], w_coef),
                         scaled_channels(channels[2], w_coef),
                         kernel_sizes[1],
                         expansion_coefs[1],
                         se_ratio,
                         strides[1],
                         scaled_repeats(repeats[1], d_coef),
                         dropout_rate=dropout_rate)
    
    stage4 = MBConvBlock(stage3, 
                         scaled_channels(channels[2], w_coef),
                         scaled_channels(channels[3], w_coef),
                         kernel_sizes[2],
                         expansion_coefs[2],
                         se_ratio,
                         strides[2],
                         scaled_repeats(repeats[2], d_coef),
                         dropout_rate=dropout_rate)
    
    stage5 = MBConvBlock(stage4, 
                         scaled_channels(channels[3], w_coef),
                         scaled_channels(channels[4], w_coef),
                         kernel_sizes[3],
                         expansion_coefs[3],
                         se_ratio,
                         strides[3],
                         scaled_repeats(repeats[3], d_coef),
                         dropout_rate=dropout_rate)

    stage6 = MBConvBlock(stage5, 
                         scaled_channels(channels[4], w_coef),
                         scaled_channels(channels[5], w_coef),
                         kernel_sizes[4],
                         expansion_coefs[4],
                         se_ratio,
                         strides[4],
                         scaled_repeats(repeats[4], d_coef),
                         dropout_rate=dropout_rate)
    
    stage7 = MBConvBlock(stage6, 
                         scaled_channels(channels[5], w_coef),
                         scaled_channels(channels[6], w_coef),
                         kernel_sizes[5],
                         expansion_coefs[5],
                         se_ratio,
                         strides[5],
                         scaled_repeats(repeats[5], d_coef),
                         dropout_rate=dropout_rate)
    
    stage8 = MBConvBlock(stage7, 
                         scaled_channels(channels[6], w_coef),
                         scaled_channels(channels[7], w_coef),
                         kernel_sizes[6],
                         expansion_coefs[6],
                         se_ratio,
                         strides[6],
                         scaled_repeats(repeats[6], d_coef),
                         dropout_rate=dropout_rate)
       
    stage9 = ConvBlock(stage8,
                       filters=scaled_channels(channels[8], w_coef),
                       kernel_size=1,
                       padding='same')
    
    stage9 = GlobalAveragePooling2D()(stage9)
    stage9 = Dense(classes, 
                       activation='softmax',
                       kernel_initializer=DENSE_KERNEL_INITIALIZER)(stage9)

    model = Model(inputs, stage9)

    return model


#channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
#channels = [16, 8, 12, 20, 40, 66, 96, 160, 640]
channels = [32, 8, 12, 20, 20, 32, 64, 96, 192]
expansion_coefs = [1, 6, 6, 6, 6, 6, 6]
repeats = [1, 2, 2, 3, 3, 4, 1]
strides = [1, 2, 2, 2, 1, 2, 1]
kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
d_coef, w_coef, r_coef, dropout_rate = efficientnet_params('efficientnet-b0')   

conv_base = EfficientNet_B0(channels,
                            expansion_coefs,
                            repeats,
                            strides,
                            kernel_sizes,
                            d_coef,
                            w_coef,
                            r_coef,
                            dropout_rate)
conv_base.summary()

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)))
model.add(conv_base)    # 1st Conv Block

model_effi = tf.keras.models.load_model("C:/Users/paur/Documents/Invernadero/Models/Efficient/Efficient_Falseunfree.h5")
import cnn_tf_models as cnn_tf 
classes = 4
epochs = 20
path_data_source = "C:/Users/paur/Documents/Invernadero/Greenhouse_project/dataset"
img_height = 224
img_width = 224
# Define some parameters for the loader:
batch_size = 32
train_data, validation_data, test_data = cnn_tf.split_tratin_test_set(path_data_source,batch_size,img_height, img_width)
from tensorflow import keras
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
    
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
# Initialize and compile distiller
distiller = Distiller(student=model, teacher=model_effi)
distiller.compile(
    optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=['acc'],
    student_loss_fn=CategoricalCrossentropy(from_logits=True),
distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=2)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)
# Distill teacher to student
distiller_hist = distiller.fit(train_data, validation_data = validation_data, epochs=1, validation_steps=len(validation_data),
              steps_per_epoch = len(train_data), callbacks=[callback] )

model.predict (test_data, verbose = 1)

model.save ("C:/Users/paur/Documents/Invernadero/Models/Efficient/tiny.h5")
