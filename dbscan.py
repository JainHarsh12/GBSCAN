import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.dbscan import dbscan
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from folium.plugins import HeatMap

# Simulate GPS Data
np.random.seed(0)
data = {
    'latitude': np.random.uniform(37.77, 37.78, 100),  # Random latitudes within a range
    'longitude': np.random.uniform(-122.43, -122.42, 100),  # Random longitudes within a range
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='T')  # Sample timestamps
}
df = pd.DataFrame(data)

# Preprocess Data
# Extract latitude and longitude as a feature matrix
X = df[['latitude', 'longitude']].values

# Set GBSCAN parameters
eps = 0.001  # Radius within which to search for neighbors (small for demo purposes)
min_samples = 5  # Minimum number of points required to form a cluster

# Run GBSCAN clustering
gbscan_instance = dbscan(X, eps, min_samples)
gbscan_instance.process()

# Extract clusters and noise
clusters = gbscan_instance.get_clusters()
noise = gbscan_instance.get_noise()

# Plot clusters on map
# Initialize map centered around the mean of latitudes and longitudes
center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Add clusters to the map
for cluster in clusters:
    cluster_points = X[cluster]
    HeatMap(cluster_points).add_to(m)

# Add noise points
for point in X[noise]:
    folium.CircleMarker(location=[point[0], point[1]], radius=3, color="red", fill=True).add_to(m)

# Save the map as an HTML file
m.save('traffic_hotspot_map.html')
print("Map saved as 'traffic_hotspot_map.html'")

# Visualize clusters on a static scatter plot
plt.figure(figsize=(10, 6))
for i, cluster in enumerate(clusters):
    cluster_points = X[cluster]
    plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i + 1}')

# Plot noise points
noise_points = X[noise]
plt.scatter(noise_points[:, 1], noise_points[:, 0], color='red', label='Noise', marker='x')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Traffic Congestion Hotspots Detected by GBSCAN')
plt.show()
