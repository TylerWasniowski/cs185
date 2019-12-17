import numpy as np
import matplotlib.pyplot as plt


def cluster(data, eps, min_samples):
    data = np.array(data)

    # For efficiently querying neighbors
    # tree = KDTree(data)

    next_label = 1
    labels = np.zeros(len(data), dtype=np.int32)
    visited = set()

    for i in range(len(data)):
        points = [i]

        while points:
            point = points[0]
            if point in visited:
                del points[0]
                continue

            visited.add(point)
            neighbors = []
            for j in range(len(data)):
                dist = np.linalg.norm(data[j] - data[point])
                if dist <= eps:
                    neighbors.append(j)

            # Core point
            if len(neighbors) >= min_samples:
                # Previously unlabelled core point, add new cluster
                if labels[point] < 1:
                    labels[point] = next_label
                    next_label += 1

                # Add nearby points to cluster
                for neighbor in neighbors:
                    labels[neighbor] = labels[point]

                # Add all neighbors to points unless visited
                points.extend(filter(lambda neighbor: neighbor not in visited, neighbors))

            del points[0]

    print(np.unique(labels, return_counts=True))

    if len(data) > 0 and len(data[0]) == 2:
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.show()

    return labels


smiley_face_data = [
    [1.0, 5.0],
    [1.25, 5.35],
    [1.25, 5.75],
    [1.5, 6.25],
    [1.75, 6.75],
    [2.0, 6.5],
    [3.0, 7.75],
    [3.5, 8.25],
    [3.75, 8.75],
    [3.95, 9.1],
    [4.0, 8.5],
    [2.5, 7.25],
    [2.25, 7.75],
    [2.0, 6.5],
    [2.75, 8.25],
    [4.5, 8.9],
    [9.0, 5.0],
    [8.75, 5.85],
    [9.0, 6.25],
    [8.0, 7.0],
    [8.5, 6.25],
    [8.5, 6.75],
    [8.25, 7.65],
    [7.0, 8.25],
    [6.0, 8.75],
    [5.5, 8.25],
    [5.25, 8.75],
    [4.9, 8.75],
    [5.0, 8.5],
    [7.5, 7.75],
    [7.75, 8.25],
    [6.75, 8.0],
    [6.25, 8.25],
    [4.5, 8.9],
    [5.0, 1.0],
    [1.25, 4.65],
    [1.25, 4.25],
    [1.5, 3.75],
    [1.75, 3.25],
    [2.0, 3.5],
    [3.0, 2.25],
    [3.5, 1.75],
    [3.75, 8.75],
    [3.95, 0.9],
    [4.0, 1.5],
    [2.5, 2.75],
    [2.25, 2.25],
    [2.0, 3.5],
    [2.75, 1.75],
    [4.5, 1.1],
    [5.0, 9.0],
    [8.75, 5.15],
    [8.0, 2.25],
    [8.25, 3.0],
    [8.5, 4.75],
    [8.5, 4.25],
    [8.25, 3.35],
    [7.0, 1.75],
    [8.0, 3.5],
    [6.0, 1.25],
    [5.5, 1.75],
    [5.25, 1.25],
    [4.9, 1.25],
    [5.0, 1.5],
    [7.5, 2.25],
    [7.75, 2.75],
    [6.75, 2.0],
    [6.25, 1.75],
    [4.5, 1.1],
    [3.0, 4.5],
    [7.0, 4.5],
    [5.0, 3.0],
    [4.0, 3.35],
    [6.0, 3.35],
    [4.25, 3.25],
    [5.75, 3.25],
    [3.5, 3.75],
    [6.5, 3.75],
    [3.25, 4.0],
    [6.75, 4.0],
    [3.75, 3.55],
    [6.25, 3.55],
    [4.75, 3.05],
    [5.25, 3.05],
    [4.5, 3.15],
    [5.5, 3.15],
    [4.0, 6.5],
    [4.0, 6.75],
    [4.0, 6.25],
    [3.75, 6.5],
    [4.25, 6.5],
    [4.25, 6.75],
    [3.75, 6.25],
    [6.0, 6.5],
    [6.0, 6.75],
    [6.0, 6.25],
    [5.75, 6.75],
    [5.75, 6.25],
    [6.25, 6.75],
    [6.25, 6.25],
    [9.5, 9.5],
    [2.5, 9.5],
    [1.0, 8.0]
]
