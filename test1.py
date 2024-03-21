import pandas as pd
import numpy as np
import matplotlib as plt

def custom_kmedoids(data, medoids, distance_func):
  """
  Performs k-medoids clustering with custom medoids and distance function.

  Args:
      data: A pandas dataframe containing the data points.
      medoids: A list of data points representing the initial medoids.
      distance_func: A function that calculates the distance between two data points.

  Returns:
      A list containing cluster labels for each data point.
  """
  data_points = data.values
  n_clusters = len(medoids)

  # Initialize clusters with medoids
  clusters = [[] for _ in range(n_clusters)]
  for i, point in enumerate(data_points):
    distances = [distance_func(point, medoid) for medoid in medoids]
    cluster_label = distances.index(min(distances))
    clusters[cluster_label].append(i)

  # Iterate until convergence
  converged = False
  while not converged:
    converged = True
    new_clusters = [[] for _ in range(n_clusters)]
    for cluster_label, points in enumerate(clusters):
      for point_index in points:
        distances = [distance_func(data_points[point_index], medoid) for medoid in medoids]
        new_cluster_label = distances.index(min(distances))
        if new_cluster_label != cluster_label:
          converged = False
        new_clusters[new_cluster_label].append(point_index)
    clusters = new_clusters

  # Assign labels to data points
  labels = [-1] * len(data_points)
  for i, cluster in enumerate(clusters):
    for point_index in cluster:
      labels[point_index] = i

  return labels

# Define distance functions
def euclidean_distance(p1, p2):
  return np.linalg.norm(p1 - p2)

def manhattan_distance(p1, p2):
  return np.sum(np.abs(p1 - p2))

def chebyshev_distance(p1, p2):
  return np.max(np.abs(p1 - p2))

# Load data and medoids
# data = pd.DataFrame({
#   "f1": [0.574, 0.006, 0.481, 0.284, 0.154, 0.617, 0.259, 0.198],
#   "f2": [0.045, 1.000, 0.091, 0.045, 0.000, 0.000, 0.000, 0.000],
#   "f3": [0.186, 1.000, 0.128, 0.085, 0.043, 0.072, 0.043, 0.087]
# })

def plot_clusters(data, labels, medoids, title):
  """
  Plots the data points with labels and medoids.

  Args:
      data: A pandas dataframe containing the data points.
      labels: A list containing cluster labels for each data point.
      medoids: A list of data points representing the medoids.
      title: The title for the plot.
  """
  colors = plt.cm.get_cmap('tab10')(labels)  # Use colormap for labels

  # Plot data points with colors based on labels
  plt.scatter(data['f1'], data['f2'], c=colors, alpha=0.7, label='Data')

  # Plot medoids with different marker
  for medoid in medoids:
    plt.scatter(medoid[0], medoid[1], marker='*', s=100, c='black', label='Medoid')

  # Add labels and title
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title(title)
  plt.legend()
  plt.show()

data = pd.read_csv("dataset.csv")

medoids = [
  [0.938, 0.000, 0.046],
  [0.617, 0.000, 0.072],
  [0.198, 0.000, 0.087],
  [0.093, 0.000, 0.043],
  [0.191, 0.000, 0.004]
]

# Perform clustering for different distances and number of clusters
for n_clusters in range(2, 6):
  print(f"Clustering with {n_clusters} clusters:")
  for distance_func in [euclidean_distance, manhattan_distance, chebyshev_distance]:
    distance_name = distance_func.__name__.replace("_", " ")
    labels = custom_kmedoids(data.copy(), medoids[:n_clusters], distance_func)
    print(f"  - {distance_name}: {labels}")
