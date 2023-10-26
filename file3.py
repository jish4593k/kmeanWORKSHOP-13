import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from torchvision import transforms
from PIL import Image
from skimage import color
import cv2

# Load and preprocess the image
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    transform = transforms.Compose([transforms.Resize((250, 250)), transforms.ToTensor()])
    img = transform(img)
    img = img.permute(1, 2, 0).numpy()
    return img

# Apply K-means clustering to the image
def k_means_compression(image, num_clusters=16):
    pixel_values = image.reshape(-1, 3).astype(np.float32)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)
    centers = kmeans.cluster_centers_

    # Compress the image by replacing pixel values with cluster centers
    compressed_image = centers[labels].reshape(image.shape)

    return compressed_image, centers, labels

# Visualize the original, compressed, and cluster-colored images
def visualize_images(original, compressed, centers, labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    
    axes[1].imshow(compressed)
    axes[1].axis('off')
    axes[1].set_title('Compressed Image')
    
    labeled_image = np.zeros_like(original)
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            labeled_image[i, j] = centers[labels[i * original.shape[0] + j]]
    axes[2].imshow(labeled_image)
    axes[2].axis('off')
    axes[2].set_title('Cluster-Colored Image')

    plt.show()

if __name__ == "__main__":
    # Load and preprocess the image
    file_path = 'weixin.jpg'
    image = load_and_preprocess_image(file_path)

    # Perform K-means compression
    num_clusters = 16
    compressed_image, centers, labels = k_means_compression(image, num_clusters)

    # Visualize the original, compressed, and cluster-colored images
    visualize_images(image, compressed_image, centers, labels)
