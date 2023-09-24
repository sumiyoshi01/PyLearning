import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 人工データセットの生成
n_samples = 1500
dataset = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42)
X = dataset[0]

# K-Means モデルの作成と訓練
kmeans = KMeans(n_clusters=4, n_init=10)  # n_initを明示的にセット
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# クラスタリング結果の可視化
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()