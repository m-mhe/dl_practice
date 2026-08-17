# Fashion MNIST Image Classification Using a Multilayer Perceptron

A neural-network-based image classification project developed using
**Python, Jupyter Notebook, TensorFlow, and Keras**. The project
investigates the classification of clothing images from the **Fashion
MNIST** dataset and explores preprocessing, neural-network training,
image augmentation, model evaluation, and overfitting.

Repository: https://github.com/m-mhe/dl_practice.git

------------------------------------------------------------------------

## 1. Project Overview

The objective of this project is to develop a machine-learning model
capable of classifying grayscale images of clothing into one of ten
predefined categories.

The primary dataset used is **Fashion MNIST**, which contains 70,000
grayscale images representing common clothing and footwear items. Each
image has a resolution of **28 × 28 pixels**.

The project was implemented and documented in a **Jupyter Notebook**
using TensorFlow/Keras. In addition to training a baseline neural
network, the project investigates the effect of increasing the effective
size and diversity of the training data through **image augmentation**.

A particular focus of the augmentation process was to introduce random
patterns or pixel-level modifications **on the clothing objects rather
than indiscriminately modifying the background**. This approach was
intended to create additional variations while preserving the essential
structure and identity of the original clothing item.

------------------------------------------------------------------------

## 2. Objectives

The main objectives of the project are:

1.  To understand and implement an image-classification workflow using
    Fashion MNIST.
2.  To preprocess grayscale image data appropriately for neural-network
    training.
3.  To develop a fully connected Multilayer Perceptron (MLP) using
    TensorFlow/Keras.
4.  To train and evaluate the model using supervised learning.
5.  To investigate the effects of model complexity and training on
    generalization.
6.  To identify and analyze overfitting using training and validation
    performance.
7.  To extend the effective training dataset using image augmentation.
8.  To evaluate whether augmented images can provide additional training
    diversity beyond the original dataset.
9.  To analyze model performance using accuracy, loss curves,
    predictions, and a confusion matrix.

------------------------------------------------------------------------

## 3. Dataset

The project uses the **Fashion MNIST** dataset available through Keras.

Fashion MNIST consists of grayscale images of clothing and footwear.
Each image is associated with an integer label from 0 to 9.

### Dataset Characteristics

  Property                              Value
  -------------------------- ----------------
  Total images                         70,000
  Original training images             60,000
  Original test images                 10,000
  Image dimensions             28 × 28 pixels
  Image type                        Grayscale
  Number of classes                        10
  Pixel-value range                    0--255

For the main experiment, the available training data was divided into
training and validation portions. The primary training experiment
therefore operated on approximately **55,000 original training images**,
with the remaining portion used for validation.

The test set was kept separate from training and was used for final
evaluation.

### Fashion MNIST Classes

    Label Class
  ------- -------------
        0 T-shirt/top
        1 Trouser
        2 Pullover
        3 Dress
        4 Coat
        5 Sandal
        6 Shirt
        7 Sneaker
        8 Bag
        9 Ankle boot

------------------------------------------------------------------------

## 4. Technologies and Libraries

The project was developed using the following technologies:

-   **Python 3**
-   **Jupyter Notebook**
-   **TensorFlow**
-   **Keras**
-   **NumPy**
-   **Matplotlib**
-   **scikit-learn**

TensorFlow/Keras was used to construct, train, save, and evaluate the
neural-network model. NumPy was used for numerical manipulation and
construction of the augmented dataset, while Matplotlib and scikit-learn
were used for visualization and evaluation.

------------------------------------------------------------------------

## 5. Setup and Installation

### 5.1 Clone the Repository

The project repository is available at:

https://github.com/m-mhe/dl_practice.git

Clone it using:

``` bash
git clone https://github.com/m-mhe/dl_practice.git
cd dl_practice
```

### 5.2 Create a Virtual Environment

Using a virtual environment is recommended to isolate the project's
dependencies.

``` bash
python -m venv .venv
```

Activate the environment on Linux/macOS:

``` bash
source .venv/bin/activate
```

On Windows:

``` powershell
.venv\Scripts\activate
```

### 5.3 Install Dependencies

Install the required packages with:

``` bash
pip install tensorflow keras numpy matplotlib scikit-learn jupyter
```

If the repository contains a `requirements.txt` file, the dependencies
can instead be installed using:

``` bash
pip install -r requirements.txt
```

### 5.4 Start Jupyter Notebook

Launch Jupyter Notebook:

``` bash
jupyter notebook
```

Open the Fashion MNIST notebook and execute the cells in their intended
order.

The Fashion MNIST dataset can be downloaded automatically through Keras
when the dataset-loading code is executed for the first time.

------------------------------------------------------------------------

## 6. Project Workflow

The complete experimental workflow can be summarized as follows:

``` text
Fashion MNIST
     │
     ▼
Dataset Exploration
     │
     ▼
Train / Validation / Test Preparation
     │
     ▼
Pixel Normalization
     │
     ▼
Image Augmentation
     │
     ▼
Extended Training Dataset
     │
     ▼
MLP Model Construction
     │
     ▼
Model Training
     │
     ▼
Validation
     │
     ▼
Test Evaluation
     │
     ├──────────────► Accuracy
     │
     ├──────────────► Loss Curves
     │
     ├──────────────► Confusion Matrix
     │
     └──────────────► Sample Predictions
```

------------------------------------------------------------------------

## 7. Data Preprocessing

### 7.1 Pixel Normalization

The original Fashion MNIST images contain integer pixel values between 0
and 255.

Before being provided to the neural network, the pixel values are
normalized to the range 0--1:

``` python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

Normalization reduces the numerical scale of the input features and
generally makes optimization more stable and efficient.

The same preprocessing must be applied when the trained model is used
for inference. For example, if a new image is converted into a Fashion
MNIST-like 28 × 28 grayscale image, its pixel values should also be
normalized before prediction.

------------------------------------------------------------------------

### 7.2 Image Representation

Each Fashion MNIST image has the shape:

``` text
28 × 28
```

Since the selected model is a fully connected MLP rather than a
convolutional neural network, each image is flattened before entering
the dense layers.

Therefore:

``` text
28 × 28 = 784
```

The image can be represented as a vector of 784 numerical input
features.

The Keras `Flatten` layer performs this transformation automatically.

------------------------------------------------------------------------

## 8. Training Dataset Extension Through Image Augmentation

One of the main experimental components of this project was the
expansion of the training data through **image augmentation**.

Instead of relying solely on the approximately **55,000 original
training images**, additional training samples were generated by
applying controlled random modifications to existing images.

### 8.1 Motivation

A neural network can memorize characteristics of the training samples
when it is exposed to a limited set of examples, particularly when the
model has a relatively large number of trainable parameters.

Image augmentation provides additional variations of existing examples.
The intention is not to create completely new classes, but rather to
expose the model to plausible variations of the same objects.

This can potentially improve the diversity of the training data and
reduce the tendency of the model to rely excessively on specific pixel
patterns.

### 8.2 Object-Focused Random Pattern Augmentation

The augmentation approach used in this project introduced **random
patterns or pixel modifications on the clothing objects rather than
primarily modifying the background**.

This distinction is important.

A naive pixel-noise strategy could modify pixels throughout the entire
image, including the background. However, Fashion MNIST images generally
contain a relatively simple background surrounding the clothing item.
Randomly modifying the background could therefore introduce patterns
that are not representative of the object itself.

The implemented approach instead attempts to place the random
modifications within the region occupied by the clothing object.

Conceptually:

``` text
Original Image
      │
      ▼
Identify / use object region
      │
      ▼
Apply controlled random pattern
      │
      ▼
Generate augmented image
      │
      ▼
Add augmented image to training data
```

The purpose is to preserve the object's general structure while
introducing additional visual variation.

### 8.3 Extended Dataset

The augmented samples were combined with the original training samples
to create an extended training dataset.

Conceptually:

``` python
X_train_extended = np.concatenate(
    [X_train, augmented_data_1, augmented_data_2, ...]
)
```

The corresponding labels were extended in the same manner so that every
augmented image retained the class label of its original image.

For example:

``` text
Original image:       Shirt
Augmented image:      Shirt
Augmented image:      Shirt
Augmented image:      Shirt
```

The augmented images therefore do not introduce new class labels. They
introduce additional variations of existing examples.

### 8.4 Important Consideration

Augmentation increases the **number and diversity of training samples**,
but it does not increase the number of independent real-world examples
in the original dataset.

For this reason, the extended dataset should be described as an
**augmented training dataset** rather than an entirely new dataset.

------------------------------------------------------------------------

## 9. Model Architecture

The primary classifier is a **Multilayer Perceptron (MLP)** implemented
using Keras.

The architecture is:

``` text
Input Image
    │
    ▼
28 × 28 Grayscale Image
    │
    ▼
Flatten
    │
    ▼
Dense Layer — 300 neurons
    │
    ▼
Dense Layer — 100 neurons
    │
    ▼
Dense Layer — 10 neurons
    │
    ▼
Softmax Probabilities
    │
    ▼
Predicted Class
```

### Layer Configuration

  Layer     Configuration         Function
  --------- --------------------- -------------------------------------
  Input     `(28, 28)`            Receives the input image
  Flatten   784 values            Converts the image into a vector
  Dense     300 neurons, ReLU     Learns intermediate representations
  Dense     100 neurons, ReLU     Learns higher-level representations
  Output    10 neurons, Softmax   Produces class probabilities

A representative implementation is:

``` python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

### ReLU Activation

The hidden layers use the **Rectified Linear Unit (ReLU)** activation
function:

``` text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity into the network, allowing the model to
learn more complex relationships between input pixels and class labels.

### Softmax Output

The final layer contains 10 neurons, corresponding to the ten Fashion
MNIST classes.

The Softmax activation converts the output values into a probability
distribution:

``` text
P(class 0)
P(class 1)
...
P(class 9)
```

The class with the highest predicted probability is selected as the
final prediction.

------------------------------------------------------------------------

## 10. Model Compilation and Training

The model is trained using supervised learning.

During training, the model repeatedly performs the following process:

1.  A batch of training images is provided to the network.
2.  The network performs a forward pass.
3.  Predictions are compared with the actual labels.
4.  The loss function measures the prediction error.
5.  Backpropagation calculates gradients.
6.  The optimizer updates the model's trainable parameters.
7.  The process is repeated for subsequent batches and epochs.

### Loss Function

The model uses:

``` python
sparse_categorical_crossentropy
```

This loss function is appropriate because the target labels are integer
class identifiers rather than one-hot encoded vectors.

### Optimizer

The training experiments use **Stochastic Gradient Descent (SGD)** as
the optimization algorithm.

A representative compilation configuration is:

``` python
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["sparse_categorical_accuracy"]
)
```

The validation set is monitored during training to evaluate how well the
model generalizes beyond the data used for updating its parameters.

------------------------------------------------------------------------

## 11. Overfitting Analysis

An important observation during experimentation was the difference
between training and validation performance.

In one of the training runs, the model reached approximately:

-   **Training accuracy:** 94%
-   **Validation accuracy:** 89%

The training accuracy continued to improve while validation performance
did not improve at the same rate. This behavior is characteristic of
**overfitting**.

Overfitting occurs when a model learns the training examples too
specifically and consequently performs worse on previously unseen data.

The training and validation curves provide a useful way of identifying
this behavior.

For example:

``` text
Training performance
        ↑
        │       ─────────────
        │     /
        │   /
        │ /
        └────────────────────→ Epochs

Validation performance
        ↑
        │     ────────
        │   /
        │ /
        └────────────────────→ Epochs
```

The exact curves and values should be taken from the final notebook
execution.

------------------------------------------------------------------------

## 12. Model Evaluation

The model is evaluated using the test dataset, which remains separate
from the training process.

### Accuracy

Classification accuracy measures the proportion of correctly classified
images:

``` text
Accuracy =
Number of Correct Predictions
──────────────────────────────
Total Number of Predictions
```

Accuracy provides an overall indication of classification performance,
although it does not show which particular classes are difficult for the
model.

### Confusion Matrix

A confusion matrix provides a more detailed view of classification
behavior.

Each row represents the actual class, while each column represents the
predicted class.

This makes it possible to identify classes that the model frequently
confuses.

Fashion MNIST contains several visually similar categories, including:

-   T-shirt/top and Shirt
-   Pullover and Coat
-   Shirt and Coat
-   Sandal and Sneaker

Consequently, the confusion matrix is particularly useful for
understanding the types of errors made by the model.

------------------------------------------------------------------------

## 13. Output Preview

The notebook generates several visual outputs to make the experiment and
its results easy to inspect.

The following sections provide locations for the actual output figures
generated during the experiment.

### 13.1 Sample Dataset Images

Place a representative screenshot or exported figure here:

``` text
docs/images/sample-images.png
```

Then display it in the README:

![Sample Fashion MNIST images](docs/images/sample-images.png)

This visualization provides an immediate overview of the images and the
ten classification categories.

------------------------------------------------------------------------

### 13.2 Augmentation Examples

A particularly useful visualization for this project is a comparison
between original and augmented images.

Recommended location:

``` text
docs/images/augmentation-examples.png
```

![Image augmentation examples](docs/images/augmentation-examples.png)

This figure should demonstrate how random patterns were introduced into
the object region while retaining the overall identity and structure of
the clothing item.

------------------------------------------------------------------------

### 13.3 Training and Validation Curves

Recommended location:

``` text
docs/images/training-curves.png
```

![Training and validation curves](docs/images/training-curves.png)

The curves show how training and validation loss/accuracy changed
throughout the training process.

They can be used to analyze:

-   Learning progression
-   Convergence
-   Generalization
-   Overfitting
-   Differences between training and validation performance

------------------------------------------------------------------------

### 13.4 Confusion Matrix

Recommended location:

``` text
docs/images/confusion-matrix.png
```

![Fashion MNIST confusion matrix](docs/images/confusion-matrix.png)

The confusion matrix provides class-level information about the model's
performance.

------------------------------------------------------------------------

### 13.5 Sample Predictions

Recommended location:

``` text
docs/images/sample-predictions.png
```

![Sample predictions](docs/images/sample-predictions.png)

A prediction visualization can show the input image together with its
actual and predicted class.

For example:

``` text
Actual: Sneaker
Predicted: Sneaker
Confidence: 0.XX
```

The exact confidence value should be taken from the model's actual
output.

------------------------------------------------------------------------

## 14. Recommended Repository Structure

A clean repository structure for the project is:

``` text
dl_practice/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── fashion_mnist_classification.ipynb
│
├── models/
│   └── fashion_mnist_model.keras
│
└── docs/
    └── images/
        ├── sample-images.png
        ├── augmentation-examples.png
        ├── training-curves.png
        ├── confusion-matrix.png
        └── sample-predictions.png
```

The exact file and directory names may differ depending on the current
repository structure.

------------------------------------------------------------------------

## 15. Saving and Loading the Model

The trained model can be saved using the Keras `.keras` format:

``` python
model.save("fashion_mnist_model.keras")
```

The saved model can later be loaded with:

``` python
model = tf.keras.models.load_model("fashion_mnist_model.keras")
```

When loading the model in another notebook or application, it is
important to reproduce the same preprocessing pipeline used during
training.

For example, if the training data was normalized from 0--255 to 0--1,
new input images must undergo the same normalization before prediction:

``` python
image = image / 255.0
```

Using inconsistent preprocessing can result in a significant decrease in
prediction accuracy even when the model itself has not changed.

------------------------------------------------------------------------

## 16. Reproducing the Experiment

To reproduce the project:

1.  Clone the repository.
2.  Install the required Python dependencies.
3.  Launch Jupyter Notebook.
4.  Open the Fashion MNIST notebook.
5.  Execute the dataset-loading and preprocessing cells.
6.  Run the image-augmentation cells to generate the extended training
    data.
7.  Construct the MLP model.
8.  Compile the model using the selected loss function and optimizer.
9.  Train the model while monitoring validation performance.
10. Evaluate the final model using the test dataset.
11. Generate the accuracy/loss curves.
12. Generate the confusion matrix.
13. Generate sample predictions.
14. Save the trained model if required.

------------------------------------------------------------------------

## 17. Limitations

Although the MLP provides a useful baseline for Fashion MNIST
classification, it has several limitations.

### Fully Connected Architecture

The model flattens every 28 × 28 image into a 784-element vector. This
removes the explicit two-dimensional spatial structure of the image.

A pixel's location relative to neighboring pixels is important for image
recognition, but a standard dense network does not exploit spatial
locality as effectively as a convolutional architecture.

### Similar Clothing Classes

Several Fashion MNIST classes have similar visual characteristics. This
makes some classification errors difficult to avoid, particularly
between categories such as Shirt, T-shirt/top, Coat, and Pullover.

### Synthetic Augmentation

The augmented images are derived from existing images rather than
independently collected real-world examples. Therefore, augmentation
increases training diversity but does not provide the same information
as collecting additional independent data.

### Generalization to Real Photographs

Fashion MNIST images are standardized 28 × 28 grayscale images. Real
photographs contain substantially more variation in lighting,
background, scale, orientation, texture, and clothing shape.

Consequently, a model trained exclusively on Fashion MNIST should not be
expected to classify arbitrary real-world clothing photographs
accurately without an appropriate preprocessing and domain-adaptation
pipeline.

------------------------------------------------------------------------

## 18. Future Improvements

Several improvements could be explored in future experiments:

-   Replace the MLP with a **Convolutional Neural Network (CNN)**.
-   Compare **SGD** with **Adam** and other optimizers.
-   Perform systematic hyperparameter tuning.
-   Experiment with different numbers of hidden layers and neurons.
-   Introduce dropout and other regularization techniques.
-   Use early stopping to reduce unnecessary training and overfitting.
-   Compare multiple augmentation strategies.
-   Measure per-class precision, recall, and F1-score.
-   Investigate class-specific errors using the confusion matrix.
-   Test different learning rates and batch sizes.
-   Evaluate the model on appropriately preprocessed external clothing
    images.
-   Compare the performance of the original and augmented training
    datasets under identical evaluation conditions.

------------------------------------------------------------------------

## 19. Conclusion

This project presents an end-to-end image-classification experiment
using the Fashion MNIST dataset and a TensorFlow/Keras-based Multilayer
Perceptron.

The work covers the complete machine-learning pipeline, including
dataset exploration, preprocessing, normalization, model construction,
supervised training, validation, test evaluation, prediction, and
performance visualization.

A significant component of the project is the use of **object-focused
image augmentation** to extend the effective training data beyond the
approximately 55,000 original training samples used in the experiment.
Random patterns were introduced primarily within the clothing-object
region rather than indiscriminately modifying the background. This
provided additional variations of existing examples while attempting to
preserve their class identity.

The project also demonstrates the practical challenge of
**overfitting**. Comparing training and validation performance provides
insight into how a neural network can achieve substantially better
performance on training data than on unseen validation data.

Overall, the project serves as a practical study of neural-network-based
image classification and provides a foundation for further investigation
using convolutional neural networks, improved regularization,
alternative optimization algorithms, and more advanced augmentation
techniques.

------------------------------------------------------------------------

## Author

**MMH EMON**\
Department of Computer Science and Engineering\
East Delta University

------------------------------------------------------------------------

## Repository

**GitHub:** https://github.com/m-mhe/dl_practice.git
