import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from torchvision import transforms
from PIL import Image
from skimage import color

# Load the image and preprocess it
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    transform = transforms.Compose([transforms.Resize((250, 250)), transforms.ToTensor()])
    img = transform(img)
    img = img.permute(1, 2, 0).numpy()
    return img

# Apply K-means clustering to the image
def k_means_segmentation(image, num_clusters=16):
    pixel_values = image.reshape(-1, 3).astype(np.float32)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)
    centers = kmeans.cluster_centers_

    # Reshape the labels to match the image shape
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image, centers

# Visualize the segmented image
def visualize_segmentation(segmented_image, centers):
    segmented_image_color = np.zeros(segmented_image.shape + (3,), dtype=np.uint8)

    for i in range(len(centers)):
        segmented_image_color[segmented_image == i] = np.uint8(centers[i])

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(segmented_image, cmap='tab20', vmin=0, vmax=19)
    plt.axis('off')
    plt.title('Segmented Image')
    plt.subplot(122)
    plt.imshow(segmented_image_color)
    plt.axis('off')
    plt.title('Segmented Image with Colors')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess the image
    file_path = 'weixin.jpg'
    image = load_and_preprocess_image(file_path)

    # Perform K-means segmentation
    num_clusters = 16
    segmented_image, centers = k_means_segmentation(image, num_clusters)

    # Visualize the segmented image
    visualize_segmentation(segmented_image, centers)
