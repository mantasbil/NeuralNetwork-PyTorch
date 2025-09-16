# Neural Networks using PyTorch - Fashion MNIST Classification

This notebook demonstrates the implementation and training of a simple neural network model using PyTorch for the classification of the Fashion MNIST dataset.

## Overview

The notebook covers the following steps:

1.  **Data Importing and Preprocessing**: Cloning the Fashion MNIST dataset from a GitHub repository, loading the training and test data, and performing initial preprocessing (normalization).
2.  **Data Loading Optimization**: Implementing a custom PyTorch `Dataset` class (`FashionMNIST`) to efficiently load data samples dynamically from disk, which is beneficial for larger datasets. This class also handles the stratified splitting of the data into training, validation, and test sets.
3.  **Dataset Loading with DataLoader**: Instantiating PyTorch `DataLoader` objects for the training, validation, and test datasets to handle batching and shuffling.
4.  **Sequential Neural Network Model**: Defining a simple sequential neural network architecture with two hidden layers using `torch.nn.Sequential`.
5.  **Neural Network Training**: Implementing the training loop, including the use of the Adam optimizer, a linear learning rate scheduler, and early stopping based on validation loss.
6.  **Model Saving and Loading**: Saving and loading the trained model's state dictionary.

## Dataset

The Fashion MNIST dataset consists of 70,000 grayscale images of size 28x28 pixels, categorized into 10 classes of clothing items. The dataset is split into 60,000 training images and 10,000 test images.

The 10 classes are:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Model Architecture

The neural network model has the following structure:

-   Input layer: 784 features (28x28 flattened image)
-   Hidden Layer 1: 300 neurons with ReLU activation and dropout
-   Hidden Layer 2: 200 neurons with ReLU activation and dropout
-   Output layer: 10 neurons (for the 10 classes) with Softmax activation

## Training

The model is trained using:

-   **Optimizer**: Adam with momentum
-   **Learning Rate Scheduler**: Linear scheduler
-   **Loss Function**: CrossEntropyLoss
-   **Early Stopping**: Based on validation loss with a specified patience.

## Results

The training history, including training and validation loss, and validation F1 score, is recorded and plotted to visualize the model's performance during training.

## How to Use

To run this notebook:

1.  Ensure you have the necessary libraries installed (PyTorch, NumPy, scikit-learn, matplotlib, pandas).
2.  Run the cells sequentially. The data will be automatically downloaded and preprocessed.
3.  The model will be trained, and the training history will be displayed and plotted.
4.  The trained model's state dictionary will be saved.
