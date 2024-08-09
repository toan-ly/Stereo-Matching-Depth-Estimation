import numpy as np

def distance(x, y, method='L1'):
    if method == 'L1':
        return abs(x - y)
    if method == 'L2':
        return (x - y) ** 2
    raise ValueError('Method unsupported!')

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
