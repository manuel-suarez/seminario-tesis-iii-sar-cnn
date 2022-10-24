import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

import os
from skimage.util import random_noise
import sys
import time
from tqdm.notebook import tqdm
import shutil

# Prepare data
INPUT_SIZE= (64,64)
BS=16
ROOT_DIR="/home/est_posgrado_manuel.suarez"

DATASET=os.path.join(ROOT_DIR,'data/sentinel12/v_2')
DATA_GEN_INPUT=os.path.join(ROOT_DIR,'DATASET')

if os.path.exists(DATA_GEN_INPUT):
    shutil.rmtree(DATA_GEN_INPUT)
os.mkdir(DATA_GEN_INPUT)

src=os.path.join(DATASET,"agri/s2")
dst=os.path.join(DATA_GEN_INPUT,"DATA")
os.symlink(src,dst)

def preprocessing_function(img):
    return np.float32(img/127.5-1)

generator=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocessing_function)
train_generator=generator.flow_from_directory(DATA_GEN_INPUT,
                                              target_size=INPUT_SIZE,
                                              class_mode=None,
                                              color_mode='grayscale',
                                              batch_size=BS,
                                              follow_links=True,)

plt.figure(figsize=(5,5))
plt.imshow(next(train_generator)[0],cmap='gray')
plt.colorbar()

def test_model(data_generator):
    img1,img2=next(data_generator)[:2]
    noise_var=np.random.rand()*.25
    # noise_var=.3
    noisy_img1=random_noise(img1,mode='speckle',var=noise_var,clip=True)
    noisy_img2=random_noise(img2,mode='speckle',var=noise_var,clip=True)
    noisy_img1=np.expand_dims(noisy_img1,axis=[0,-1])
    noisy_img2=np.expand_dims(noisy_img2,axis=[0,-1])
    denoised_img1=model.predict(noisy_img1)
    denoised_img2=model.predict(noisy_img2)
    fig,ax=plt.subplots(3,2,figsize=(10,12))
    mapple=ax[0,0].imshow(img1)
    plt.colorbar(mapple,ax=ax[0,0])
    mapple=ax[0,1].imshow(img2)
    plt.colorbar(mapple,ax=ax[0,1])
    mapple=ax[1,0].imshow(noisy_img1[0].reshape(INPUT_SIZE))
    plt.colorbar(mapple,ax=ax[1,0])
    mapple=ax[1,1].imshow(noisy_img2[0].reshape(INPUT_SIZE))
    plt.colorbar(mapple,ax=ax[1,1])
    mapple=ax[2,0].imshow(denoised_img1[0].reshape(INPUT_SIZE))
    plt.colorbar(mapple,ax=ax[2,0])
    mapple=ax[2,1].imshow(denoised_img2[0].reshape(INPUT_SIZE))
    plt.colorbar(mapple,ax=ax[2,1])
    plt.savefig("figure1.png")
    #plt.show()

# Model creation
def create_model(input_shape=(256, 256, 1)):
    # Input Layer
    input_layer = Input(shape=input_shape, name="InputLayer")
    # Layer 1 (Conv+ReLU)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name="Layer1_Conv2D")(input_layer)
    x = ReLU(name="Layer1_ReLU")(x)
    #  Layer 2-7 (Conv+BN+ReLU)
    for i in range(2, 8):
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name=f"Layer{i}_Conv2D")(x)
        x = BatchNormalization(name=f"Layer{i}_BN")(x)
        x = ReLU(name=f"Layer{i}_ReLU")(x)
    # Layer 8 (Conv+ReLU)
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', name="Layer8_Conv2D")(x)
    x = ReLU(name="Layer8_ReLU")(x)
    # Division residual layer
    x = tf.math.divide(input_layer, x)
    # Nonlinear function layer
    x = tf.math.tanh(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)

# Define custom loss
mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction='none')
l_tv = l_tv=0.002
def id_cnn_loss_fn(y_true, y_pred):
    mse = tf.reduce_sum(mse_loss_fn(y_true, y_pred))
    variational_loss = tf.image.total_variation(y_pred)

    total_loss = mse + l_tv * variational_loss
    return tf.reduce_mean(total_loss)

model=create_model(list(INPUT_SIZE)+[1])
model.summary()

test_model(train_generator)

EPOCHS = 100 # The paper has trained the model for 2000 epochs
lr=2e-3

max_var=.3

# Compile
model.compile(loss=id_cnn_loss_fn, optimizer=Adam(learning_rate=2e-5, beta_1=0.5))

# Train
model.fit(train_generator, epochs=EPOCHS)