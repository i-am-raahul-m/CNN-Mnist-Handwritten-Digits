import numpy as np
import struct
import matplotlib.pyplot as plt


def load_mnist_images(file_path, flattened=False):
    with open(file_path, 'rb') as f:
        # Read file header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for image file")
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        if (flattened):
            images = images.reshape(num_images, rows*cols)
        else:
            images = images.reshape(num_images, rows, cols)
    return images


def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read file header
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for label file")
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


if __name__ == '__main__':
    # Paths to the MNIST IDX files
    origin_path = ""
    images_path = origin_path + "dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    labels_path = origin_path + "dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

    # Load the data
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)

    # Print shapes
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # View image, label
    sample_no = 9999
    plt.imshow(images[sample_no].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {labels[sample_no]}")
    plt.show()
