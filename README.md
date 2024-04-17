# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries . 
2.Read the data frame using pandas. 
3.Get the information regarding the null values present in the dataframe. 
4.Apply label encoder to the non-numerical column inoreder to convert into numerical values. 
5.Determine training and test data set. 6.Apply k means clustering for customer segmentation

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Prasannalakshmi G
RegisterNumber: 212222240075

```
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers_EX8.csv")
data

X = data[['Annual Income (k$)','Spending Score (1-100)']]
X

plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors = ['r', 'g', 'b', 'c', 'm']
for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], 
              color=colors[i], label=f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Output:
## DATASET:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118610231/188edcdd-9e6e-470c-a82e-9e0f9a75d6e8)

## GRAPH:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118610231/f0c6925e-4e78-4bb3-ad8d-188ffbaf1dec)

## K-Means:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118610231/51fd4867-0b64-4bb9-bed6-f0b0b92d8911)


## CENTROID VALUE:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118610231/00dc1988-e0d3-4dd5-aace-750011ef4d2a)

## K-Means CLUSTERING:
![image](https://github.com/Prasannalakshmiganesan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118610231/7283f9a3-d67d-45c6-8730-9451a6c1132e)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
