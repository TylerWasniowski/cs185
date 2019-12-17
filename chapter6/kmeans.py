import numpy as np
import matplotlib.pyplot as plt


def cluster(data, k, iters=150):
    labels = np.zeros(len(data), dtype=np.int32)

    np.random.shuffle(data)
    clusters = np.array_split(data, k)

    centers = []
    for it in range(iters):
        # Flip cluster so each column is a row, compute average of each row, use that as the value for the centroid
        centers = [[np.sum(y)/len(y) for y in np.transpose(x)] for x in clusters]

        for i in range(len(data)):
            min_dist = np.inf
            for j in range(len(centers)):
                if len(centers[j]) > 0:
                    # Euclidean distance
                    dist = np.linalg.norm(np.subtract(centers[j], data[i]))
                    if dist < min_dist:
                        min_dist = dist
                        labels[i] = j

        new_clusters = [[] for _ in range(len(clusters))]
        for i in range(len(data)):
            new_clusters[labels[i]].append(data[i])
        clusters = new_clusters

    if len(data) > 0 and len(data[0]) == 2:
        plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=labels)
        plt.show()

    return clusters, centers


old_faithful_data = [
    [3.600, 79],
    [1.800, 54],
    [2.283, 62],
    [3.333, 74],
    [2.883, 55],
    [4.533, 85],
    [1.950, 51],
    [1.833, 54],
    [4.700, 88],
    [3.600, 85],
    [1.600, 52],
    [4.350, 85],
    [3.917, 84],
    [4.200, 78],
    [1.750, 62],
    [1.800, 51],
    [4.700, 83],
    [2.167, 52],
    [4.800, 84],
    [1.750, 47]
]
