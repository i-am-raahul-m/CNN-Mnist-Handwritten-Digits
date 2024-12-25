import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from math import ceil
from mnist_viewer import load_mnist_images, load_mnist_labels

### GPU COMPUTE
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Device used: {device}")


### PATHS
origin_path = ""
model_num = 0
model_file_path = origin_path + f"models/cnn{model_num}.pth"
images_path = origin_path + "dataset/train-images-idx3-ubyte/train-images-idx3-ubyte"
labels_path = origin_path + "dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
train_stats_path = origin_path + f"training_statistics/train_stats_cnn{model_num}.pkl"


### CONSTANTS
dataset_size = 60000
in_channels = 1
img_size = 28
input_shape = (img_size, img_size)
output_dim = 10


### HYPER-PARAMETERS
learning_rate = 0.001
batch_size = 32
epochs = 30
num_of_batches = ceil(dataset_size / batch_size)


### CNN ARCHITECTURE
class CNN(nn.Module):
    def __init__(self, in_channels, input_shape, output_dim):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # (28, 28, 1)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # (28, 28, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (14, 14, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (14, 14, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (7, 7, 64)

            nn.Flatten(),
            
            nn.Linear((input_shape[0]//4*input_shape[1]//4)*64, 128),
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
    for i in range(1, output_dim):
        if output_tensor[max_ind] < output_tensor[i]:
            max_ind = i
    return max_ind

# label --> tensor
def label2tensor(label):
    array = np.zeros(output_dim)
    array[label] = 1.0
    return torch.tensor(array, dtype=torch.float32, device=device)

# Dataset image input tensor creation
def input_image_tensor(batch_idx, batch_size):
    offset = batch_idx*batch_size
    image_np_array = images[offset:offset+batch_size]
    tensor = torch.tensor(image_np_array, dtype=torch.float32, device=device).unsqueeze(1)
    tensor /= 255
    return tensor

# Dataset Label input tensor creation
def input_label_tensor(batch_idx, batch_size):
    offset = batch_idx*batch_size
    label_np_array = labels[offset:offset+batch_size]
    # label_np_array = np.vectorize(label2tensor)(label_np_array)  # One-hot encoding (not required)
    tensor = torch.tensor(label_np_array, device=device)
    return tensor


### CNN TRAINING CONTEXT
# Models
cnn = CNN(in_channels=in_channels, input_shape=input_shape, output_dim=output_dim).to(device=device)

# Loss Function
loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer_cnn = optim.Adam(cnn.parameters(), lr=learning_rate)


### TRAINING THE MODEL
# Loading entire dataset
images = load_mnist_images(images_path, flattened=False)
labels = load_mnist_labels(labels_path)

# Training Loop
cnn_loss_history = np.zeros(epochs)

for epoch in range(epochs):
    avg_cnn_loss = 0
    for batch_idx in range(num_of_batches):
        image_tensor = input_image_tensor(batch_idx, batch_size)
        label_tensor = input_label_tensor(batch_idx, batch_size)

        optimizer_cnn.zero_grad()
        output = cnn(image_tensor)
        cnn_loss = loss_function(output, label_tensor)
        cnn_loss.backward()
        optimizer_cnn.step()

        avg_cnn_loss += cnn_loss
    avg_cnn_loss /= num_of_batches

    # Storing loss values after every epoch
    cnn_loss_val = avg_cnn_loss.item()
    print(f"Completed: {epoch}/{epochs}  Loss: {cnn_loss_val: .4f}")
    cnn_loss_history[epoch] = cnn_loss_val


### SAVING MODEL WEIGHTS and STATISTICS
# Save built and trained mlp model
torch.save(cnn.state_dict(), model_file_path)

# Save training statistics
with open(train_stats_path, "wb") as file:
    pickle.dump(cnn_loss_history, file)
