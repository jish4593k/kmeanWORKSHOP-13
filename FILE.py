import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from torchvision import transforms
from PIL import Image

# Load the image and preprocess it
def load_image(file_path):
    img = Image.open(file_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img.view(-1, 3)
    return img

# Apply K-means clustering to the image
def k_means_segmentation(image, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(image)
    labels = kmeans.predict(image)
    centers = kmeans.cluster_centers_
    segmented_image = labels.reshape(image.shape[0], image.shape[1])
    return segmented_image, centers

# Visualize the segmented image
def visualize_segmentation(segmented_image, centers):
    segmented_image = (segmented_image / (len(centers) - 1) * 255).astype(np.uint8)
    colored_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
    for i in range(len(centers)):
        colored_image[segmented_image == i] = (int(centers[i][0]), int(centers[i][1]), int(centers[i][2]))
    plt.imshow(colored_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main":
    # Load the image
    file_path = 'weixin.jpg'
    image = load_image(file_path)

    # Perform K-means segmentation
    num_clusters = 2
    segmented_image, centers = k_means_segmentation(image.numpy(), num_clusters)

    # Visualize the segmented image
    visualize_segmentation(segmented_image, centers)
