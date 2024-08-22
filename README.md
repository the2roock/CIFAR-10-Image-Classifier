# Cifar10-Classification

## Project Overview
This project focuses on a classification task using the CIFAR-10 dataset, a popular benchmark for image classification. The goal is to develop and evaluate a deep learning model to accurately classify images into one of the 10 predefined categories. The final model achieved an accuracy of **88.7%**.


## Required Skills & Tools
To understand and work with this project, the following skills and tools are required:

- **Skills**:
  - Data Engineering
  - Data Mining
  - Classification Techniques
  - Data Visualization
  - Neural Networks
  - Convolutional Neural Networks(CNNs)
  - Residual Networks(ResNets)

- **Tools**:
  - Python
  - NumPy
  - Pandas
  - TensorFlow
  - Scikit-Learn
  - Matplotlib


# Data Description
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. The classes represent common objects like airplanes, cars, birds, etc.

- **Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

- **Features Preprocessing:**
  - Normalization of pixel values to the [0, 1] range.

- **Augmentation:**
  - Random rotation of images within a range of -15 to 15 degrees.
  - Width and height shifting by 10% of the image size.
  - Horizontal flipping of images.

- **Expected Outputs:** Labels are One-Hot encoded.


## Model Overview
The model is a deep convolutional neural network (CNN) designed to effectively capture the features necessary for classification in the CIFAR-10 dataset. Below is an overview of the architecture:

![Model Architecture](path/to/architecture_image.png)

- **Layers**:
  - **Input Layer**: Accepts 32x32 RGB images.
  - **Conv2D Layers**: Multiple convolutional layers with increasing filters (64, 128, 256) to capture spatial features.
  - **Pooling Layers**: MaxPooling2D and AveragePooling2D layers to reduce the spatial dimensions of the feature maps.
  - **Add Layers**: Implements skip connections to enhance learning, resembling residual networks.
  - **Batch Normalization**: Used after each add layer to stabilize and speed up training.
  - **Dense Layers**: Fully connected layers for classification.
  
- **Architecture Details**:
  - **Conv2D Layers**: 
    - First block: Conv2D with 64 filters, followed by BatchNormalization and MaxPooling.
    - Second block: Conv2D with 128 filters, combined with skip connections and MaxPooling.
    - Third block: Conv2D with 256 filters, incorporating additional skip connections and MaxPooling.
  - **Batch Normalization**: Applied after each skip connection to normalize the output.
  - **Final Layer**: Fully connected (Dense) layer with a softmax activation for output.
  
- **Parameter Count**:
  - **Total Trainable Parameters**: 2,229,248

- **Special Components**:
  - **Residual Connections**: Skip connections are used to prevent vanishing gradients and improve learning in deeper networks.
  - **Batch Normalization**: Helps in normalizing the input to each layer, thus speeding up the training process and making the model more stable.


# Test Results

# Use Case
