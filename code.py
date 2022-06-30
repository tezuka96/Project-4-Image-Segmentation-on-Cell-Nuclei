# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:58:54 2022

@author: Hilman

Project 4
Image Segmentation on Cell Nuclei
"""

#Import packages
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os, cv2, glob, datetime
from scipy import io
import numpy as np

#1. Load the dataset
#1.1 Prepare file directory
train_file_dir = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-4-Image-Segmentation-on-Cell-Nuclei\data-science-bowl-2018\train"
test_file_dir = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-4-Image-Segmentation-on-Cell-Nuclei\data-science-bowl-2018\test"

#%%
#1.2 Load the images list
def load_images(file_path):
    images=[]
    for image_file in os.listdir(file_path):
        img = cv2.imread(os.path.join(file_path,image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        images.append(img)
    return images

train_image_dir = os.path.join(train_file_dir,'inputs')
train_images = load_images(train_image_dir)
test_image_dir = os.path.join(test_file_dir,'inputs')
test_images = load_images(test_image_dir)

#1.3 Load the masks list
def load_masks(file_path):
    masks=[]
    for mask_file in os.listdir(file_path):
        mask = cv2.imread(os.path.join(file_path,mask_file),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(128,128))
        masks.append(mask)
    return masks

train_mask_dir = os.path.join(train_file_dir,'masks')
train_masks = load_masks(train_mask_dir)
test_mask_dir = os.path.join(test_file_dir,'masks')
test_masks = load_masks(test_mask_dir)

#%%   
#1.4 Convert the lists into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
#1.5. Check some examples
#for images
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

#for masks
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()  

#%%
#2. Data preprocessing
#2.1. Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
#Check the mask output
print(np.unique(train_masks[0]))

#%%
#2.2. Change the mask value (Encode into numerical encoding, binary 1 and 0)
converted_masks_train = np.ceil(train_masks_np_exp/255)
converted_masks_test = np.ceil(test_masks_np_exp/255)
converted_masks_train = 1 - converted_masks_train
converted_masks_test = 1 - converted_masks_test

#%%
#2.3. Normalize the images
converted_images_train = train_images_np / 255.0
converted_images_test = test_images_np/255.0

#%%
#2.4. Do train-validation split
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_val,y_train,y_val = train_test_split(converted_images_train,converted_masks_train,test_size=0.2,random_state=SEED)

#%%
#2.5. Convert the numpy array into tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)
test_x = tf.data.Dataset.from_tensor_slices(converted_images_test)
test_y = tf.data.Dataset.from_tensor_slices(converted_masks_test)

#%%
#2.6. Zip the tensor slice into ZipDataset
train = tf.data.Dataset.zip((train_x,train_y))
val = tf.data.Dataset.zip((val_x,val_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#2.7. Convert into PrefetchDataset
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE

train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)

val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size=AUTOTUNE)

test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
#3. Create the model
#3.1 Use a pretrained as feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#8.2. Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128,128,3])
    #Applying functional API
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x,skip])
    
    #This is the last layer of the model (output layer)
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
model.summary()

#Use in iPython console 
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#Create functions to show predictions

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

#7. Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train.take(2):
    sample_image,sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])
        
show_predictions()

#%%
#Create a callback to help to display results during training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

#Tensorboard and EarlyStopping callbacks
base_log_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-4-Image-Segmentation-on-Cell-Nuclei\tb_logs"
log_dir = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '__Project_4')
tb = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1,profile_batch=0)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=0)

#%%
#4. Start to do training
EPOCH = 200

history = model.fit(train,epochs=EPOCH,steps_per_epoch=STEPS_PER_EPOCH,batch_size=BATCH_SIZE,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=val,
                    callbacks=[DisplayCallback(),tb,es])

#%%
#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")

#%%
#Deploy model
show_predictions(test,3)

#%%
from numba import cuda 
device = cuda.get_current_device()
device.reset()