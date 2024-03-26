# Importing necessary libraries
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Function to plot customer locations
def plot_customers(data_file):
    # Read data from CSV file
    data = pd.read_csv(data_file)
    
    # Extracting Annual Income and Spending Score columns
    annual_income = data['Annual Income (k$)']
    spending_score = data['Spending Score (1-100)']
    
    # Plotting the customers
    plt.figure(figsize=(10, 6))
    plt.scatter(annual_income, spending_score, s=100)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Spending Behavior')
    plt.grid(True)
    plt.show()

# Function to perform clustering on customers
def cluster_customers(data_file, k):
    # Read data from CSV file
    data = pd.read_csv(data_file)
    
    # Extracting Annual Income and Spending Score columns
    X = data.iloc[:, [3, 4]].values
    
    # Performing KMeans clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    
    # Visualizing the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Clusters')
    plt.grid(True)
    plt.show()

# Plotting customer locations
plot_customers('shoppingdata.csv')

# Clustering customers with KMeans
cluster_customers('shoppingdata.csv', k=3)
