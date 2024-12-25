# MNIST Handwritten Digits Classification with CNN

This repository contains a Python implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The model is built, trained, tested, and used for predictions in a straightforward workflow, making it ideal for beginners and enthusiasts.

## Features

- **Dataset**: Uses the MNIST handwritten digits dataset (60,000 training images and 10,000 testing images).
- **Model Architecture**: A CNN with 2 convolutional feature-extraction layers and 2 fully connected feature-assimilation layers
- **Training**: Model is trained using standard supervised learning techniques.
- **Evaluation**: Achieves a reasonably high accuracy of ~(99+)% on the test dataset.
- **Prediction**: Can predict handwritten digits from custom input images.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- `numpy`
- `pytorch`
- `pillow`
- `matplotlib`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/i-am-raahul-m/CNN-Mnist-Handwritten-Digits.git
   cd CNN-Mnist-Handwritten-Digits
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training the Model

Run the training script to train the CNN:

```bash
python train_CNN.py
```

The script will:
- Load the MNIST dataset.
- Normalize and preprocess the images.
- Build and train the CNN model.
- Save the trained model to a file (`models/cnn_.pth`).

### 2. Testing the Model

Evaluate the model's accuracy on the test dataset:

```bash
python test_CNN.py
```

This script loads the saved model and computes accuracy performance metric on the test dataset.

### 3. Making Predictions

Use the trained model to predict custom images:

```bash
python predict_CNN.py 
```

---

Specify the path of the image file to be predicted in the "image_path" global variable

### **Model Architecture**

- **Input Layer**: 
  - Input shape: (28, 28, 1) grayscale image.

- **Convolutional Layer 1**: 
  - **Filter Size**: \(3 \times 3\)
  - **Number of Filters**: 32
  - **Stride**: 1
  - **Padding**: 1 (same padding)
  - **Output Shape**: (28, 28, 32)
  - **Activation Function**: Leaky ReLU
  - **Batch Normalization**: Yes

- **Max Pooling Layer 1**: 
  - **Filter Size**: \(2 \times 2\)
  - **Stride**: 2
  - **Output Shape**: (14, 14, 32)

- **Convolutional Layer 2**:
  - **Filter Size**: \(3 \times 3\)
  - **Number of Filters**: 64
  - **Stride**: 1
  - **Padding**: 1 (same padding)
  - **Output Shape**: (14, 14, 64)
  - **Activation Function**: Leaky ReLU
  - **Batch Normalization**: Yes

- **Max Pooling Layer 2**:
  - **Filter Size**: \(2 \times 2\)
  - **Stride**: 2
  - **Output Shape**: (7, 7, 64)

- **Flatten Layer**: 
  - Converts 3D feature maps into a 1D vector. 
  - Flattened Shape: \(7 \times 7 \times 64 = 3136\)

- **Fully Connected Layer 1**: 
  - **Number of Neurons**: 128
  - **Activation Function**: Leaky ReLU

- **Fully Connected Layer 2 (Output Layer)**: 
  - **Number of Neurons**: Based on output classes (e.g., 10 for MNIST digits).
  - **Output Shape**: (Number of output classes)
  - **Logits**: Final output without activation for classification.

### **Forward Pass**
The forward pass applies each of these layers in sequence to produce the final logits for classification.

---

### Sample Predictions

Here is a sample prediction made by the model:

|     Input Image     | Predicted Label |
|---------------------|-----------------|
| ![handwritten_digit](handwritten_digit.png) |     Digit 8      |

---

## Technologies Used

- PyTorch
- NumPy
- Matplotlib
- Python Imaging Library (PIL)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
