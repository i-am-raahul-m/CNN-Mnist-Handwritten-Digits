import numpy as np
import torch
import torch.nn as nn
from mnist_viewer import load_mnist_images, load_mnist_labels

### GPU COMPUTE
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")


### PATHS
origin_path = ""
model_num = 0
model_file_path = origin_path + f"models/cnn{model_num}.pth"
images_path = origin_path + "dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
labels_path = origin_path + "dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"


### CONSTANTS
test_dataset_size = 10000
img_size = 28
data_dim = img_size * img_size
num_classes = 10   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


### CNN ARCHITECTURE
class CNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # (28, 28, 1)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (28, 28, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (14, 14, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (14, 14, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (7, 7, 64)

            nn.Flatten(),
            
            nn.Linear(input_shape[0]*input_shape[1]*64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
            # Logits
        )

    def forward(self, x):
        return self.model(x)


### TENSOR CREATION and CONVERTION 
# tensor --> label
def tensor2label(output_tensor):
    max_ind = 0
    for i in range(1, num_classes):
        if output_tensor[max_ind] < output_tensor[i]:
            max_ind = i
    return max_ind

# label --> tensor
def label2tensor(label):
    array = np.zeros(num_classes)
    array[label] = 1.0
    return torch.tensor(array, dtype=torch.float32, device=device)

# Dataset image input tensor creation
def input_image_tensor(test_idx):
    image_tensor = torch.tensor(images[test_idx], dtype=torch.float32, device=device)
    image_tensor /= 255
    return image_tensor.unsqueeze(0)

# Dataset Label input tensor creation
def input_label(test_idx):
    label_tensor = torch.tensor(labels[test_idx], device=device)
    # label_np_array = np.vectorize(label2tensor)(label_np_array)  # One-hot encoding (not required)
    return label_tensor


### LOADING TESTING CONTEXT
# Loading entire test dataset
images = load_mnist_images(images_path, flattened=False)
labels = load_mnist_labels(labels_path)

# Loading saved model
cnn = CNN(input_dim=data_dim, output_dim=num_classes).to(device=device)
cnn.load_state_dict(torch.load(model_file_path, weights_only=True, map_location=device))
cnn.eval()


### TESTING LOOP
accurate_prediction_count = 0
for test_idx in range(test_dataset_size):
    image_tensor = input_image_tensor(test_idx)
    label = input_label(test_idx)
    
    prediction = tensor2label(cnn(image_tensor).squeeze(0))
    if (prediction == label):
        accurate_prediction_count += 1

print(f"Accuracy: {accurate_prediction_count / test_dataset_size * 100: .4f}")
