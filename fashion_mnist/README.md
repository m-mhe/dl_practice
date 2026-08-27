# Fashion MNIST Image Classification Using CNN

A deep-learning project for classifying clothing and footwear images

using the **Fashion MNIST** dataset and a fully connected **Multilayer

Perceptron (MLP)** implemented with **TensorFlow/Keras**.

The project was developed and experimented with in **Jupyter Notebook**.

In addition to the standard Fashion MNIST training data, the project

investigates the use of image augmentation to create a substantially

larger training set from the original training samples. The final model

is also evaluated on manually prepared real-world clothing images,

including **Sandal** and **Pant/Trouser** examples.

**Repository:** https://github.com/m-mhe/dl_practice.git

------------------------------------------------------------------------

## 1. Project Overview

The purpose of this project is to investigate how a neural network can

learn to classify grayscale clothing images into the ten categories

provided by the Fashion MNIST dataset.

The project follows a complete supervised machine-learning workflow:

1.  Load the Fashion MNIST dataset.

2.  Explore and visualize the dataset.

3.  Prepare the training, validation, and test data.

4.  Normalize image pixel values.

5.  Extend the training dataset using image augmentation.

6.  Build a Convolutional Neural Network (CNN).

7.  Train the model using supervised learning.

8.  Monitor training and validation performance.

9.  Investigate overfitting and generalization.

10. Save the trained model in Keras format.

11. Evaluate the saved model separately.

12. Test the model on manually prepared clothing images.

13. Analyze the model's predictions and classification performance.

The project therefore goes beyond simply obtaining an accuracy score. It

examines the complete process of preparing image data, training a neural

network, controlling overfitting, evaluating the resulting model, and

applying the trained model to images outside the original dataset.

------------------------------------------------------------------------

# 2. Dataset

The project uses the **Fashion MNIST** dataset provided through Keras.

Fashion MNIST is a dataset of grayscale images representing different

categories of clothing and footwear. Each image has a resolution of **28

× 28 pixels**.

## Dataset Characteristics

Property                                Value

---------------------------- ----------------

Total images                           70,000

Original training images               60,000

Original test images                   10,000

Image dimensions               28 × 28 pixels

Image channels                  1 (grayscale)

Number of classes                          10

Original pixel-value range             0--255

For the main training experiment, the original training set was divided

so that approximately **55,000 images** were used as the primary

training data, while a separate portion was used for validation.

The official Fashion MNIST test set remained separate from the training

process and was used for evaluating generalization on unseen dataset

examples.

------------------------------------------------------------------------

## 3. Fashion MNIST Classes

The dataset contains ten classes:

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

The model's final layer contains ten output neurons, with each neuron

corresponding to one of these classes.

------------------------------------------------------------------------

# 4. Technologies Used

The project was developed using:

-   **Python**

-   **Jupyter Notebook**

-   **TensorFlow**

-   **Keras**

-   **NumPy**

-   **Matplotlib**

-   **scikit-learn**

### Main responsibilities of the libraries

**TensorFlow/Keras**

Used for constructing, compiling, training, evaluating, and saving the

neural-network models.

**NumPy**

Used for manipulating image arrays, preparing labels, and generating and

combining augmented training data.

**Matplotlib**

Used for visualizing Fashion MNIST images, training progress,

predictions, and other experimental results.

**scikit-learn**

Used for evaluation and analysis, including classification-related

evaluation tools such as the confusion matrix.

------------------------------------------------------------------------

# 5. Repository Structure

The current project is organized inside the `fashion_mnist` directory.

``` text

dl_practice/

│

└── fashion_mnist/

│

├── README.md

├── main.ipynb

├── evaluate.ipynb

├── fashion\_mnist\_model.keras

├── final\_model.keras

├── realWorld.jpg

├── realWorldOne.jpg

├── x.jpg

└── y.jpg

```

## File Descriptions

### `main.ipynb`

This is the primary research and development notebook.

It contains the main experimentation process, including dataset

preparation, preprocessing, augmentation, model development, training,

and analysis.

### `evaluate.ipynb`

This notebook is used for model evaluation and testing.

It contains the evaluation process for the trained Fashion MNIST model

and includes testing the model on manually prepared images.

### `fashion_mnist_model.keras`

A saved Keras model produced during the Fashion MNIST training

experiments.

### `final_model.keras`

The final trained model selected for the project.

This model is used for the final evaluation and external image

prediction experiments.

### `realWorld.jpg`

A manually prepared real-world clothing image used to evaluate the model

outside the original Fashion MNIST test dataset.

### `realWorldOne.jpg`

Another manually prepared real-world clothing image used for external

evaluation.

### `x.jpg`

A test image used during the external prediction/evaluation process.

### `y.jpg`

Another test image used during the external prediction/evaluation

process.

------------------------------------------------------------------------

# 6. Setup

## 6.1 Clone the Repository

Clone the repository using:

``` bash

git clone https://github.com/m-mhe/dl_practice.git

cd dl_practice

```

Navigate to the Fashion MNIST project:

``` bash

cd fashion_mnist

```

## 6.2 Create a Virtual Environment

Creating a virtual environment is recommended:

``` bash

python -m venv .venv

```

On Linux/macOS:

``` bash

source .venv/bin/activate

```

On Windows:

``` powershell

.venv\Scripts\activate

```

## 6.3 Install Dependencies

Install the required packages:

``` bash

pip install tensorflow keras numpy matplotlib scikit-learn jupyter

```

If a `requirements.txt` file is added to the project, it can instead be

installed with:

``` bash

pip install -r requirements.txt

```

## 6.4 Start Jupyter Notebook

Run:

``` bash

jupyter notebook

```

Open:

``` text

main.ipynb

```

to reproduce the main experiment.

Open:

``` text

evaluate.ipynb

```

to perform the evaluation and external image testing.

The Fashion MNIST dataset is downloaded automatically by Keras when the

dataset-loading code is executed for the first time.

------------------------------------------------------------------------

# 7. Data Preprocessing

## 7.1 Image Format

Fashion MNIST images are grayscale images with dimensions:

``` text

28 × 28

```

Each pixel contains an integer value between 0 and 255.

A value close to 0 represents a dark pixel, while a value close to 255

represents a bright pixel.

------------------------------------------------------------------------

## 7.2 Pixel Normalization

Before training, the pixel values are normalized from the original

range:

``` text

0–255

```

to:

``` text

0–1

```

using:

``` python

X_train = X_train / 255.0

X_test = X_test / 255.0

```

Normalization is important because neural-network optimization generally

works more effectively when input features are placed on a consistent

and relatively small numerical scale.

The same normalization must also be applied to images supplied to the

model during inference.

For example:

``` python

image = image / 255.0

```

If the model is trained using normalized images but an external image is

passed using unnormalized 0--255 pixel values, prediction performance

can decrease substantially.

------------------------------------------------------------------------

# 8. Training and Validation Data

The original Fashion MNIST training set contains 60,000 images.

For the main experiment, approximately **55,000 images** were used as

the primary training set, with a separate portion retained for

validation.

The validation data allows the model to be evaluated on examples that

are not used to update its parameters during training.

This distinction is important:

``` text

Training data

↓

Used to update model weights

Validation data

↓

Used to monitor generalization during training

Test data

↓

Used for final evaluation

```

Keeping these roles separate helps provide a more reliable estimate of

how well the model generalizes.

------------------------------------------------------------------------

# 9. Training Dataset Extension Through Image Augmentation

One of the significant parts of this project is the use of **image

augmentation** to extend the effective training dataset beyond the

approximately 55,000 original training samples.

Instead of training exclusively on the original images, additional

variations were generated from existing training examples.

The purpose was to provide the neural network with more diverse examples

while preserving the original class identity.

------------------------------------------------------------------------

## 9.1 Why Augmentation Was Used

A neural network with a relatively large number of trainable parameters

can potentially memorize characteristics of the training samples.

This becomes especially important when the model reaches very high

training accuracy while validation accuracy stops improving.

Image augmentation can provide additional variations of existing

examples.

The fundamental idea is:

``` text

Original training image

      ↓

Random controlled modification

      ↓

Augmented training image

      ↓

Add to training dataset

```

The augmented image retains the label of the original image.

For example:

``` text

Original image       → Shirt

Augmented image #1   → Shirt

Augmented image #2   → Shirt

Augmented image #3   → Shirt

```

------------------------------------------------------------------------

# 10. Object-Focused Random Pattern Augmentation

The augmentation used in this project was not intended to simply add

random noise across the entire image.

Instead, the experiment focused on adding random patterns or pixel-level

modifications **to the object region rather than primarily modifying the

background**.

This distinction is important because Fashion MNIST contains a

relatively simple background around the clothing object.

A completely unrestricted random-noise operation could introduce

artificial patterns into the background that are unrelated to the

clothing itself.

The project therefore attempted to make the augmentation more relevant

to the object being classified.

Conceptually:

``` text

Original Fashion MNIST image

      │

      ▼

Determine / use object region

      │

      ▼

Apply random pattern

      │

      ▼

Generate modified clothing image

      │

      ▼

Preserve original class label

      │

      ▼

Add to extended training set

```

The goal was to expose the model to additional variations while

maintaining the fundamental characteristics required to identify the

clothing category.

------------------------------------------------------------------------

# 11. Constructing the Extended Dataset

The generated augmented samples were combined with the original training

samples.

Conceptually:

``` python

X_train_extended = np.concatenate([

X\_train,

augmented\_data\_1,

augmented\_data\_2,

...

])

```

The labels were extended correspondingly:

``` python

y_train_extended = np.concatenate([

y\_train,

y\_augmented\_1,

y\_augmented\_2,

...

])

```

Each augmented image retains the class label of the original image from

which it was generated.

This means the augmentation process does **not** introduce additional

Fashion MNIST classes. Instead, it increases the number and diversity of

training examples available to the model.

It is therefore more precise to describe the resulting collection as an

**augmented training dataset**, rather than as an entirely new

independent dataset.

------------------------------------------------------------------------

# 12. Model Architecture

The final model is a **Convolutional Neural Network (CNN)** designed for
28 × 28 grayscale Fashion MNIST images.

The architecture is:

Input: 28 × 28 × 1
        │
        ▼
Data Augmentation
├── Random Rotation: ±7.5°
├── Random Translation:
│   ├── Height: ±7.5%
│   └── Width: ±7.8%
└── Random Zoom: ±8.5%
        │
        ▼
Conv2D: 128 filters, 3 × 3, ReLU, same padding
        │
        ▼
MaxPooling2D: 2 × 2
        │
        ▼
Conv2D: 128 filters, 3 × 3, ReLU, same padding
        │
        ▼
MaxPooling2D: 2 × 2
        │
        ▼
Conv2D: 128 filters, 3 × 3, ReLU, same padding
        │
        ▼
Flatten
        │
        ▼
Dense: 128 neurons + ReLU
        │
        ▼
Dense: 10 neurons + Softmax
        │
        ▼
Fashion MNIST Class

The actual model definition is:

# --------------------------------------------------
# 2nd Data augmentation
# --------------------------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(
        7.5 / 360.0
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.075,
        width_factor=0.078
    ),

    tf.keras.layers.RandomZoom(
        height_factor=(-0.085, 0.085),
        width_factor=(-0.085, 0.085)
    )
])


# --------------------------------------------------
# CNN
# --------------------------------------------------

CNNModel = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    augmentation,

    tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


CNNModel.summary()

The model contains three convolutional layers followed by pooling layers,
a flattening operation, one hidden dense layer, and a ten-neuron output layer.
The convolutional layers learn spatial features such as edges, shapes, and
textures, while the final dense layers perform classification.

The model's parameter count depends on the exact Keras architecture and should
be taken directly from CNNModel.summary() rather than estimated from a
different architecture.

------------------------------------------------------------------------

# 13. Data Augmentation Layer

The CNN includes Keras preprocessing layers directly in the model. These
layers apply augmentation during training to expose the network to controlled
variations of the training images.

The augmentation configuration is:

Augmentation

Configuration

Random Rotation

±7.5°

Random Translation

Height ±7.5%, Width ±7.8%

Random Zoom

±8.5%

The augmentation layer is active during training but is automatically inactive
during validation, evaluation, and prediction. Therefore, validation and test
images remain unaugmented.

Conceptually:

Training image
      │
      ▼
Random augmentation
      │
      ▼
CNN
      │
      ▼
Prediction

During validation or testing:

Validation/Test image
      │
      ▼
Augmentation layer inactive
      │
      ▼
CNN
      │
      ▼
Prediction

This allows the model to learn from augmented training examples while
evaluating generalization on the original, unmodified validation and test
images.

------------------------------------------------------------------------

# 14. Convolutional Feature Extraction

The CNN uses three convolutional layers:

Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")

Each convolutional layer contains 128 learnable filters. A 3 × 3 kernel
allows the network to examine local neighborhoods of the image and learn
spatial patterns.

The same padding preserves the spatial dimensions before pooling.

Two max-pooling layers reduce the spatial resolution:

MaxPooling2D(pool_size=(2, 2))

The feature-map progression is:

28 × 28 × 1
     ↓
28 × 28 × 128
     ↓ MaxPool
14 × 14 × 128
     ↓
14 × 14 × 128
     ↓ MaxPool
 7 × 7 × 128
     ↓
 7 × 7 × 128
     ↓ Flatten
6272 features

The resulting feature representation is then passed to the dense classifier.

------------------------------------------------------------------------

# 15. Output Layer

The final layer is:

``` python

Dense(10, activation=tf.keras.activations.softmax)

```

There are ten output neurons because Fashion MNIST contains ten classes.

Softmax converts the output values into probabilities.

Conceptually:

``` text

T-shirt/top   → 0.xx

Trouser       → 0.xx

Pullover      → 0.xx

Dress         → 0.xx

Coat          → 0.xx

Sandal        → 0.xx

Shirt         → 0.xx

Sneaker       → 0.xx

Bag           → 0.xx

Ankle boot    → 0.xx

```

The class with the highest probability becomes the model's predicted

class.

------------------------------------------------------------------------

# 16. Model Training

The model is trained using supervised learning.

For each training image, the model receives:

``` text

Input image → Correct class label

```

During each training iteration:

1.  The image is passed through the network.

2.  The network generates a probability distribution.

3.  The prediction is compared with the true label.

4.  The loss is calculated.

5.  Backpropagation calculates gradients.

6.  The optimizer updates the model parameters.

7.  The process continues for subsequent batches.

The process is repeated across multiple epochs.

------------------------------------------------------------------------

# 17. Loss Function

The project uses **Sparse Categorical Cross-Entropy** for

classification.

``` python

loss="sparse_categorical_crossentropy"

```

This loss function is suitable when the target labels are represented as

integer class IDs such as:

``` text

0, 1, 2, ..., 9

```

rather than one-hot encoded vectors.

The loss measures how far the model's predicted probability distribution

is from the correct class.

A lower loss indicates that the model's predictions are becoming more

consistent with the true labels.

------------------------------------------------------------------------

# 18. Optimizer

The training experiment uses **Stochastic Gradient Descent (SGD)**.

SGD updates the model's trainable parameters using gradients calculated

from training batches.

Conceptually:

``` text

Weights

↓

Forward pass

↓

Calculate loss

↓

Backpropagation

↓

Calculate gradients

↓

SGD weight update

↓

New weights

```

The process continues until the model has completed the specified number

of training epochs.

------------------------------------------------------------------------

# 19. Overfitting Investigation

Overfitting was one of the important issues observed during the project.

During experimentation, the model could achieve very high performance on

the training data while validation performance remained significantly

lower.

For example, one training run reached approximately:

``` text

Training accuracy:   ~94%

Validation accuracy: ~89%

```

The difference demonstrates that performance on the training data does

not necessarily represent performance on unseen data.

A model can memorize training examples rather than learning

representations that generalize well.

The project therefore experimented with techniques such as:

-   Increasing the effective size of the training data through

augmentation.

-   Applying dropout.

-   Monitoring validation performance.

-   Comparing training and validation curves.

-   Adjusting the model architecture.

The final architecture contains five hidden dense layers and dropout

after each hidden layer.

------------------------------------------------------------------------

# 20. Evaluation

The project uses multiple forms of evaluation rather than relying on a

single accuracy value.

Evaluation includes:

-   Training accuracy

-   Validation accuracy

-   Test accuracy

-   Training loss

-   Validation loss

-   Confusion matrix

-   Individual predictions

-   External real-world image predictions

This provides a more complete picture of the model's behavior.

------------------------------------------------------------------------

# 21. Accuracy and Loss Curves

The training history can be visualized using accuracy and loss curves.

### Accuracy Curve

The accuracy curve compares the model's performance on training and

validation data over successive epochs.

This helps determine whether the model is continuing to generalize as

training progresses.

### Loss Curve

The loss curve shows how the prediction error changes over time.

A typical overfitting pattern can occur when:

``` text

Training loss     ↓ continues decreasing

Validation loss   ↓ initially decreases

              ↑ later begins increasing

```

This indicates that the model continues to fit the training data while

its generalization performance begins to deteriorate.

------------------------------------------------------------------------

# 22. Confusion Matrix

A confusion matrix provides a class-by-class breakdown of the model's

predictions.

The rows represent the actual classes, while the columns represent the

predicted classes.

For example:

``` text

             Predicted

          0  1  2  3 ... 9

Actual     0

       1

       2

       3

      ...

       9

```

The diagonal represents correctly classified examples.

Off-diagonal values indicate classification errors.

This is particularly useful for Fashion MNIST because several classes

are visually similar.

Commonly challenging categories include:

-   T-shirt/top vs. Shirt

-   Pullover vs. Coat

-   Shirt vs. Coat

-   Sandal vs. Sneaker

The confusion matrix therefore provides information that overall

accuracy alone cannot provide.

------------------------------------------------------------------------

# 23. External Real-World Image Testing

An additional part of the project was testing the trained model on

images that were **not directly taken from the Fashion MNIST dataset**.

This experiment was designed to investigate whether a model trained on

standardized Fashion MNIST images could make meaningful predictions when

given manually prepared clothing images.

The evaluation notebook uses images such as:

``` text

realWorld.jpg

realWorldOne.jpg

x.jpg

y.jpg

```

These images were processed into a format compatible with the trained

model before prediction.

------------------------------------------------------------------------

# 24. Sandal and Pant/Trouser Testing

The model was specifically tested using manually prepared examples

representing **Sandal** and **Pant/Trouser**.

The model correctly classified these test examples.

This is an important demonstration because these images were not simply

additional entries from the original Fashion MNIST test set. They were

used as external examples to investigate how the trained model behaved

when presented with clothing imagery outside the standard dataset.

The successful predictions demonstrated that, after appropriate

preprocessing and conversion to the model's expected input format, the

model was capable of correctly identifying these tested categories.

However, these successful examples should not be interpreted as proof

that the model can reliably classify arbitrary real-world clothing

photographs. Real-world images can differ substantially from Fashion

MNIST in resolution, lighting, background, viewpoint, scale, texture,

and object appearance.

------------------------------------------------------------------------

# 25. Model Saving

The trained model can be saved using Keras's native `.keras` format:

``` python

model.save("fashion_mnist_model.keras")

```

The project repository contains:

``` text

fashion_mnist_model.keras

final_model.keras

```

These files allow the trained model to be reused without retraining from

the beginning.

------------------------------------------------------------------------

# 26. Loading the Trained Model

A saved model can be loaded using:

``` python

model = tf.keras.models.load_model("final_model.keras")

```

After loading the model, input images must undergo the same

preprocessing used during training.

For example:

``` python

image = image / 255.0

```

The input must also have the expected dimensions:

``` text

28 × 28

```

If an external photograph is used, it must first be converted into an

appropriate grayscale 28 × 28 representation before being passed to the

model.

------------------------------------------------------------------------

# 27. Why Consistent Preprocessing Matters

The model learns from the numerical representation of its training data.

During training:

``` text

Pixel value: 0–255

    ↓

Normalization

    ↓

Pixel value: 0–1

```

If the model is later given an image with unnormalized pixel values, the

numerical distribution of the input will be very different from what the

model learned during training.

Therefore, the same preprocessing pipeline must be applied during both

training and inference.

This is especially important when loading the `.keras` model in a

separate notebook.

------------------------------------------------------------------------

# 28. Research and Experimental Findings

The project provided several practical observations.

### 28.1 Training Performance Can Be Misleading

A model can achieve extremely high training accuracy while still

performing substantially worse on validation data.

Therefore, training accuracy alone is insufficient for determining

whether a model generalizes well.

### 28.2 Validation Performance Is Important

Validation accuracy and validation loss provide a better indication of

whether the model is learning generalizable patterns.

### 28.3 Model Capacity Must Be Controlled

Adding more layers and neurons increases the model's representational

capacity, but it can also increase the risk of overfitting.

The final model therefore uses dropout extensively.

### 28.4 Data Augmentation Provides Additional Variation

The augmentation experiment increases the number of training samples and

introduces additional variations derived from existing examples.

This can help the model encounter more diverse input patterns during

training.

### 28.5 External Testing Is Different From Dataset Testing

Correctly classifying a Fashion MNIST test image and correctly

classifying a manually prepared real-world image are different

evaluation scenarios.

Fashion MNIST images have a standardized appearance, while external

photographs introduce additional variation.

The successful Sandal and Pant/Trouser tests therefore provide an

interesting demonstration of external inference, but they should not

replace systematic test-set evaluation.

------------------------------------------------------------------------

# 29. Limitations

## 29.1 MLP Architecture

The model is a fully connected network rather than a CNN.

Flattening a 28 × 28 image into 784 independent input features means the

architecture does not explicitly exploit the spatial relationships

between neighboring pixels.

CNNs are generally better suited to image data because convolutional

layers can learn spatially local features such as edges, shapes, and

textures.

## 29.2 Synthetic Augmentation

The augmented images are derived from existing training samples.

Therefore, although the extended dataset contains more samples, it does

not contain the same amount of independent information as a dataset

containing additional real-world images.

## 29.3 Fashion MNIST Domain

Fashion MNIST images are standardized, low-resolution, grayscale images.

Real photographs can contain:

-   Complex backgrounds

-   Different lighting conditions

-   Shadows

-   Multiple objects

-   Different viewing angles

-   Different image resolutions

-   Different object scales

-   Textures and patterns not present in Fashion MNIST

Therefore, external-image performance may be significantly different

from Fashion MNIST test accuracy.

## 29.4 Limited External Testing

The successful Sandal and Pant/Trouser predictions demonstrate that the

model can correctly classify the tested examples, but a small number of

manually selected images is not sufficient to establish general

real-world classification performance.

------------------------------------------------------------------------

# 30. Future Improvements

The following improvements could be explored in future work:


-   Compare SGD with **Adam** and other optimizers.

-   Perform systematic hyperparameter tuning.

-   Experiment with different learning rates.

-   Compare different batch sizes.

-   Investigate early stopping.

-   Experiment with different dropout rates.

-   Compare different augmentation techniques.

-   Perform more extensive external-image testing.

-   Calculate precision, recall, and F1-score for each class.

-   Analyze class-specific errors using the confusion matrix.

-   Build a complete preprocessing pipeline for real-world photographs.

-   Investigate transfer learning using a model designed for natural

images.

-   Compare the original training dataset against the augmented training

dataset under identical conditions.

------------------------------------------------------------------------

# 31. Reproducing the Experiment

To reproduce the main experiment:

``` bash

git clone https://github.com/m-mhe/dl_practice.git

cd dl_practice/fashion_mnist

```

Install the dependencies:

``` bash

pip install tensorflow keras numpy matplotlib scikit-learn jupyter

```

Start Jupyter:

``` bash

jupyter notebook

```

Then:

1.  Open `main.ipynb`.

2.  Load the Fashion MNIST dataset.

3.  Perform the preprocessing steps.

4.  Prepare the approximately 55,000-image training set.

5.  Generate the augmented training samples.

6.  Combine the original and augmented data.

7.  Construct `modelTwo`.

8.  Compile the model.

9.  Train the model.

10. Monitor training and validation performance.

11. Evaluate the trained model.

12. Save the final model if required.

For external evaluation:

1.  Open `evaluate.ipynb`.

2.  Load `final_model.keras`.

3.  Prepare the external images.

4.  Apply the same preprocessing pipeline.

5.  Convert the images to the expected 28 × 28 grayscale format.

6.  Run model predictions.

7.  Compare the predictions with the actual classes.

------------------------------------------------------------------------

# 32. Project Structure at a Glance

``` text

dl_practice/

└── fashion_mnist/

│

├── README.md

│

├── main.ipynb

│       └── Main research, preprocessing,

│           augmentation, model training,

│           and experimentation

│

├── evaluate.ipynb

│       └── Model evaluation and external

│           image prediction

│

├── fashion\_mnist\_model.keras

│       └── Saved Fashion MNIST model

│

├── final\_model.keras

│       └── Final trained model

│

├── realWorld.jpg

│       └── External clothing test image

│

├── realWorldOne.jpg

│       └── External clothing test image

│

├── x.jpg

│       └── External evaluation image

│

└── y.jpg

        └── External evaluation image

```

------------------------------------------------------------------------

# 33. Conclusion

This project presents a complete deep-learning workflow for Fashion

MNIST image classification using a fully connected Multilayer Perceptron

implemented with TensorFlow and Keras.

The work begins with the preparation and normalization of Fashion MNIST

images and continues through model construction, supervised training,

validation, evaluation, augmentation, and external prediction.

A central component of the project is the extension of the approximately

55,000-image training set through **object-focused image augmentation**.

Random patterns were introduced primarily within the clothing-object

region instead of indiscriminately modifying the background. The

objective was to increase training diversity while preserving the

identity and fundamental structure of the original clothing examples.

The final MLP contains five hidden dense layers with 400, 300, 300, 200,

and 200 neurons respectively. ReLU activations are used in the hidden

layers, while a ten-neuron Softmax layer performs the final

classification. Dropout with a rate of 0.45 is applied after each hidden

layer as a regularization mechanism to reduce overfitting.

The project also investigates the difference between training and

validation performance, demonstrating the practical importance of

monitoring generalization rather than relying solely on training

accuracy.

Finally, the trained model was evaluated not only on Fashion MNIST data

but also on manually prepared external clothing images. In particular,

the model correctly classified the tested **Sandal** and

**Pant/Trouser** examples after the images were appropriately processed

for the Fashion MNIST input format.

Overall, the project demonstrates the complete lifecycle of a

neural-network image-classification experiment, from data preparation

and augmentation to model training, evaluation, model persistence, and

external inference.

------------------------------------------------------------------------

# 34. Author

**MMH EMON**

Department of Computer Science and Engineering\

East Delta University

------------------------------------------------------------------------

# 35. Repository

GitHub repository:

https://github.com/m-mhe/dl_practice.git
