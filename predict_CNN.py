import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

### GPU COMPUTE
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")


### PATHS
origin_path = ""
model_num = 0
model_file_path = origin_path + f"models/cnn{model_num}.pth"
image_path = origin_path + "handwritten_digit.png"


### CONSTANTS
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


### TENSOR CONVERTION 
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


### LOADING PREDICTION CONTEXT
# Loading saved model
cnn = CNN(input_dim=data_dim, output_dim=num_classes).to(device=device)
cnn.load_state_dict(torch.load(model_file_path, weights_only=True, map_location=device))
cnn.eval()

# Loading and pre-processing prediction image 
sample_image = Image.open(image_path)
grayscale_image = sample_image.convert("L")
image_array = np.array(grayscale_image).flatten()
image_tensor = torch.tensor(image_array, dtype=torch.float32, device=device).unsqueeze(0)
normalized_image_tensor = image_tensor / 255


### PREDICTION
prediction = tensor2label(cnn(normalized_image_tensor).squeeze(0))

# Visualizing prediction
plt.imshow(image_array.reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {prediction}")
plt.show()
