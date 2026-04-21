# import numpy as np
from collections import deque
import math

def convert_list_to_str(lst):
    """将二维整型list转换为(x, x), (x, x)格式的字符串"""
    return ', '.join(f'({x[0]}, {x[1]})' for x in lst)

def save_to_file(content, filename='output.txt'):
    """将内容保存到文件"""
    with open(filename, 'w') as f:
        f.write(content)

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        DBSCAN聚类算法实现
        
        参数:
        eps: 邻域半径
        min_samples: 核心点所需的最小邻域样本数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        """
        对数据集X进行聚类
        
        参数:
        X: 输入数据，形状为(n_samples, n_features)
        """
        n_samples = len(X)
        self.labels_ = [-1] * n_samples  # 初始化为-1（噪声点）
        cluster_id = 0
        
        # 计算距离矩阵（优化性能）
        distances = self._compute_distances(X)
        
        for i in range(n_samples):
            if self.labels_[i] != -1:  # 已经分类的点跳过
                continue
                
            # 找到当前点的邻域点
            neighbors = self._find_neighbors(distances, i)
            
            if len(neighbors) < self.min_samples:
                # 标记为噪声点
                self.labels_[i] = -1
            else:
                # 创建新簇
                self._expand_cluster(X, distances, i, neighbors, cluster_id)
                cluster_id += 1
        
        return self
    
    def _compute_distances(self, X):
        """计算所有点之间的距离矩阵"""
        n_samples = len(X)
        distances = [[0 for _ in range(n_samples)] for _ in range(n_samples)]
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = math.sqrt((X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2)
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def _find_neighbors(self, distances, point_idx):
        """找到指定点的邻域点"""
        res = []
        for idx, dist in enumerate(distances[point_idx]):
            if dist <= self.eps:
                res.append(idx)
        return res
    
    def _expand_cluster(self, X, distances, point_idx, neighbors, cluster_id):
        """扩展簇"""
        queue = deque([point_idx])
        self.labels_[point_idx] = cluster_id
        
        while queue:
            current_point = queue.popleft()
            current_neighbors = self._find_neighbors(distances, current_point)
            
            if len(current_neighbors) >= self.min_samples:
                for neighbor in current_neighbors:
                    if self.labels_[neighbor] == -1:  # 噪声点
                        self.labels_[neighbor] = cluster_id
                        queue.append(neighbor)
                    elif self.labels_[neighbor] == -1:  # 未分类点
                        self.labels_[neighbor] = cluster_id

# 测试示例
if __name__ == "__main__":
    # 创建测试数据
    # from sklearn.datasets import make_moons, make_blobs, make_circles
    # X_moons, _ = make_moons(n_samples=100, noise=0.05)
    # X_circles, _ = make_circles(n_samples=100, noise=0.01,factor=0.1)
    # X_blobs, _ = make_blobs(n_samples=100, centers=4, cluster_std=0.6)

    data_ori = input().split(")")
    data_ori = [d for d in data_ori if d != ""]
    eps, min_samples = input().split(" ")
    eps = float(eps)
    min_samples = int(min_samples)
    data = [[0 for _ in range(3)] for _ in range(len(data_ori))]

    for i, slice in enumerate(data_ori):
        if slice == "": continue
        idx = slice.find('(') + 1
        x, y = slice[idx:].split(",")
        data[i][0] = float(x)
        data[i][1] = float(y)
        data[i][2] = i
    # data = X_blobs.tolist()

    for i, slice in enumerate(data):
        data[i].append(i)
    
    # data_str = convert_list_to_str(data)
    # print(data_str)
    # save_to_file(data_str)
    # eps = 1
    # min_samples=3

    data = sorted(data, key=lambda x: (x[0],x[1]))
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    labels = [0] * len(data)
    for i in range(len(labels)):
        labels[data[i][2]] = dbscan.labels_[i]
    
    for d in labels:
        print(d, end=" ")
    # dbscan.fit(X_blobs)
    # print(dbscan.labels_)
    # X = data
    # unique_labels = set(dbscan.labels_)
    # try:
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     plt.figure(figsize=(10, 6))
    #     colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    #     X = np.array(X)
    #     labels_ = np.array(dbscan.labels_)
    #     for i, label in enumerate(unique_labels):
    #         if label == -1:
    #             cluster_points = X[labels_ == label]
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
    #                         c='black', marker='x', label='噪声点')
    #         else:
    #             cluster_points = X[labels_ == label]
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
    #                         c=colors[i % len(colors)], label=f'簇{label}')
        
    #     plt.title('DBSCAN聚类结果')
    #     plt.xlabel('特征1')
    #     plt.ylabel('特征2')
    #     plt.legend()
    #     plt.show()
    # except ImportError:
    #     print("如需可视化，请安装matplotlib: pip install matplotlib")