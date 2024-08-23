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

![Model Architecture](https://github.com/the2roock/Cifar10-Classification/blob/main/report_images/Cifar10%20NN%20Architecture.png)

The network employs a series of convolutional layers, residual connections, and max pooling operations to extract and learn features from the input images. The architecture includes the following key components:
1. **Input Layer**:
  - The input layer accepts images of size 32×32×3 (width, height, channels).
2. **Convolutional Blocks**:
  - The network is structured into blocks, each containing convolutional layers, residual connections, batch normalization, and activation functions.
  - The first block starts with a 64-filter convolutional layer and progresses through deeper layers with increasing filter sizes 128, 256 and 512.
3. **Residual Connections**:
  - To facilitate better gradient flow and mitigate the vanishing gradient problem, residual connections are used within blocks, allowing the network to learn identity mappings.
4. **Pooling Layers**:
  - Max and Average Pooling layers are applied after each block to reduce the spatial dimensions of the feature maps, effectively downsampling the input while retaining the most crucial features.
5. **Fully Connected Layers**:
  - After the final convolutional layer, the feature maps are flattened and passed through a series of fully connected layers, culminating in the final output layer with 10 neurons, corresponding to the 10 classes in the CIFAR-10 dataset.
  - The final layer uses a softmax activation function to output class probabilities.
6. **Parameter Count**:
  - The total number of trainable parameters in this model is detailed in the summary found in the [Cifar10_Classification.ipynb](https://github.com/the2roock/Cifar10-Classification/blob/main/Cifar10_Classification.ipynb) notebook, which includes weights and biases across all layers.
7. **Special Components**:
  - Batch Normalization: Applied after residual connections to stabilize the learning process.
  - Residual and Skip Connections: Allow the network to learn residual functions, improving the flow of gradients through the network.

![Detailed Layer Architecture](https://github.com/the2roock/Cifar10-Classification/blob/main/report_images/Cifar10%20NN%201%20Layer%20Architecture.png)
The network consists of several convolutional blocks, each designed to extract features and gradually downsample the input image. The blocks are connected through residual and skip connections.


## Training Results
### Training History
The model was trained using the **Adam** optimization algorithm with a **Categorical Cross-Entropy** loss function. The training was conducted over three stages with varying epochs and learning rates:
- **Batch Size**: 64
- **Epochs**: 20 epochs with a learning rate of 3e-4, followed by 10 epochs with a learning rate of 1e-4, and a final 5 epochs with a learning rate of 5e-5.
During training, the model's performance was monitored to ensure convergence and optimal learning. The results, including accuracy and loss over the training epochs, are visualized in the plot below. The final test score is also provided to evaluate the model's performance on unseen data.
![Training History](https://github.com/the2roock/Cifar10-Classification/blob/main/report_images/Train%20Results.png)

### Test Score
- **Test Accuracy: 88.74 %**
- **Test Loss: 0.5771**


# Use Case
