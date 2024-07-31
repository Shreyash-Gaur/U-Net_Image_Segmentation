
---

# Image Segmentation with U-Net

## Project Overview
This project demonstrates how to build, train, and evaluate a U-Net model for semantic image segmentation. The primary objective is to predict a label for every pixel in an image from a self-driving car dataset, effectively segmenting the image into different classes.

## Table of Contents
1. [Introduction](#introduction)
2. [U-Net Architecture](#u-net-architecture)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Training the Model](#training-the-model)
6. [Model Output](#model-output)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
Semantic image segmentation is a crucial task in computer vision, especially for autonomous driving, where it is essential to understand the scene at a pixel level. U-Net, a type of Convolutional Neural Network (CNN), is specifically designed for this purpose. This project involves building and training a U-Net model to perform semantic segmentation on a CARLA self-driving car dataset.

## U-Net Architecture
U-Net is composed of two main parts: the encoder (downsampling path) and the decoder (upsampling path).

### Encoder (Downsampling Path)
The encoder consists of repeated application of two 3x3 convolutions (unpadded convolutions) each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step, we double the number of feature channels.

### Decoder (Upsampling Path)
Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.

### Model Layers
The U-Net model consists of the following layers:
```
Layer (type)                  Output Shape              Param #   
==================================================================
input_1 (InputLayer)          [(None, 192, 256, 3)]     0         
__________________________________________________________________
conv2d (Conv2D)               (None, 192, 256, 32)      896       
__________________________________________________________________
conv2d_1 (Conv2D)             (None, 192, 256, 32)      9248      
__________________________________________________________________
max_pooling2d (MaxPooling2D)  (None, 96, 128, 32)       0         
__________________________________________________________________
conv2d_2 (Conv2D)             (None, 96, 128, 64)       18496     
__________________________________________________________________
conv2d_3 (Conv2D)             (None, 96, 128, 64)       36928     
__________________________________________________________________
max_pooling2d_1 (MaxPooling2D)(None, 48, 64, 64)        0         
__________________________________________________________________
conv2d_4 (Conv2D)             (None, 48, 64, 128)       73856     
__________________________________________________________________
conv2d_5 (Conv2D)             (None, 48, 64, 128)       147584    
__________________________________________________________________
max_pooling2d_2 (MaxPooling2D)(None, 24, 32, 128)       0         
__________________________________________________________________
conv2d_6 (Conv2D)             (None, 24, 32, 256)       295168    
__________________________________________________________________
conv2d_7 (Conv2D)             (None, 24, 32, 256)       590080    
__________________________________________________________________
dropout (Dropout)             (None, 24, 32, 256)       0         
__________________________________________________________________
max_pooling2d_3 (MaxPooling2D)(None, 12, 16, 256)       0         
__________________________________________________________________
conv2d_8 (Conv2D)             (None, 12, 16, 512)       1180160   
__________________________________________________________________
conv2d_9 (Conv2D)             (None, 12, 16, 512)       2359808   
__________________________________________________________________
dropout_1 (Dropout)           (None, 12, 16, 512)       0         
__________________________________________________________________
conv2d_transpose (Conv2DTransp(None, 24, 32, 256)       1179904   
__________________________________________________________________
concatenate (Concatenate)     (None, 24, 32, 512)       0         
__________________________________________________________________
conv2d_10 (Conv2D)            (None, 24, 32, 256)       1179904   
__________________________________________________________________
conv2d_11 (Conv2D)            (None, 24, 32, 256)       590080    
__________________________________________________________________
conv2d_transpose_1 (Conv2DTran(None, 48, 64, 128)       295040    
__________________________________________________________________
concatenate_1 (Concatenate)   (None, 48, 64, 256)       0         
__________________________________________________________________
conv2d_12 (Conv2D)            (None, 48, 64, 128)       295040    
__________________________________________________________________
conv2d_13 (Conv2D)            (None, 48, 64, 128)       147584    
__________________________________________________________________
conv2d_transpose_2 (Conv2DTran(None, 96, 128, 64)       73792     
__________________________________________________________________
concatenate_2 (Concatenate)   (None, 96, 128, 128)      0         
__________________________________________________________________
conv2d_14 (Conv2D)            (None, 96, 128, 64)       73792     
__________________________________________________________________
conv2d_15 (Conv2D)            (None, 96, 128, 64)       36928     
__________________________________________________________________
conv2d_transpose_3 (Conv2DTran(None, 192, 256, 32)      18464     
__________________________________________________________________
concatenate_3 (Concatenate)   (None, 192, 256, 64)      0         
__________________________________________________________________
conv2d_16 (Conv2D)            (None, 192, 256, 32)      18464     
__________________________________________________________________
conv2d_17 (Conv2D)            (None, 192, 256, 32)      9248      
__________________________________________________________________
conv2d_18 (Conv2D)            (None, 192, 256, 32)      9248      
__________________________________________________________________
conv2d_19 (Conv2D)            (None, 192, 256, 23)      759       
==================================================================
Total params: 8,640,471
Trainable params: 8,640,471
Non-trainable params: 0
__________________________________________________________________

```

## Dataset
The dataset used in this project is the CARLA self-driving car dataset. It includes images of various driving scenes and corresponding segmentation masks, where each pixel in the mask represents a class label (e.g., road, car, pedestrian).

## Preprocessing
### Steps:
1. **Load Data:** Load the images and corresponding masks from the dataset.
2. **Normalization:** Normalize the pixel values of images to the range [0, 1].
3. **Resizing:** Resize images and masks to a consistent shape (e.g., 192x256).
4. **Data Augmentation:** Apply random transformations such as rotations, flips, and zooms to increase the diversity of the training data.

### Visualization:
Visualize sample images and their corresponding masks to understand the data distribution and structure.

## Training the Model
### Loss Function:
Use sparse categorical crossentropy loss function for pixel-wise classification.

### Training Process:
1. **Compile the Model:** Compile the U-Net model using an optimizer (e.g., Adam) and the loss function.
2. **Train the Model:** Train the model using the preprocessed training dataset. Monitor training and validation accuracy.
3. **Epochs and Batch Size:** Configure the number of epochs and batch size for training.

### Visualization:
- Plot the training and validation accuracy and loss over epochs.

![Image](/results/accuracy.png)

- Display sample predictions to assess the model's performance.
![Image](/results/output.png)
![Image(/results/output1.png)
![Image](/results/output2.png)

## Results
The results of the project include:
- Visualization of training and validation accuracy and loss over epochs.
- Sample predictions on the validation dataset to demonstrate the model's ability to segment images.
- Performance metrics indicating the model's effectiveness in semantic segmentation.

## Conclusion
The "Image Segmentation with U-Net" project successfully demonstrates the application of a U-Net model for semantic segmentation on a self-driving car dataset. The project provides a clear understanding of the U-Net architecture, preprocessing steps, and the training process. The resulting model shows promising performance in accurately segmenting images at the pixel level.

## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint arXiv:1505.04597.
- CARLA self-driving car dataset.

---
