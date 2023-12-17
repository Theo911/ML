import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

# Fetch a smaller subset of the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = shuffle(mnist.data.astype('float32') / 255.0, mnist.target.astype('int'), random_state=42)
X_subset = X[:1000]  # Use a smaller subset for faster testing

# Display the first 10 images from the dataset
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(X_subset[i].reshape(28, 28), cmap='gray')
    axes[i].axis('off')
plt.show()

# Apply k-means with k=10
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_subset)

# Display the 10 cluster centers as images
centroids = kmeans.cluster_centers_

fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(centroids[i].reshape(28, 28), cmap='gray')
    axes[i].axis('off')
plt.show()

