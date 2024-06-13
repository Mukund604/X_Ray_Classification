# Bone Fracture Classification

![](https://www.mdpi.com/sensors/sensors-22-01285/article_deploy/html/images/sensors-22-01285-g020b.png)


This project aims to develop a high-accuracy binary classification model to predict bone fractures from medical images. The model uses deep learning techniques to achieve high performance in distinguishing between fractured and non-fractured bones.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Bone fractures are a common medical condition requiring accurate and prompt diagnosis for effective treatment. This project leverages the power of deep learning to develop a binary classification model capable of identifying bone fractures with high accuracy. The model is trained and evaluated on a labeled dataset of bone images, achieving significant accuracy in predictions.

## Dataset

The dataset consists of bone images labeled as either "fractured" or "not fractured". The images are preprocessed and split into training, validation, and test sets to evaluate the model's performance.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/bone-fracture-classification.git
    cd bone-fracture-classification
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## Usage

To train and evaluate the model, run the Jupyter notebook provided in the repository:

1. Start Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Open the `98-accuracy-binary-classification-bone-fracture-2.ipynb` notebook and run the cells sequentially.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. The architecture consists of multiple convolutional layers followed by pooling layers and fully connected layers, with ReLU activations and dropout for regularization.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
## Results

The model was evaluated on the validation and test datasets, achieving the following results:

- **Validation Accuracy:** 95.42%
- **Validation Loss:** 0.2795
- **Test Accuracy:** 96.44%

The model demonstrated strong performance, correctly predicting 488 out of 506 test samples.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request
