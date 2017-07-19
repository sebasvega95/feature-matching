import numpy as np

PERMS = [[(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 2), (2, 1)],
         [(0, 1), (1, 0), (2, 2)], [(0, 1), (1, 2), (2, 0)],
         [(0, 2), (1, 0), (2, 1)], [(0, 2), (1, 1), (2, 0)]]

TRIANG_TO_P = [(0, 1), (2, 3), (4, 5)]


def angle(p, i):
    a = p[i] - p[(i + 1) % 3]
    b = p[i] - p[(i + 2) % 3]
    return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def dist_angles(p, q, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return (abs(np.sin(angle(p, i1)) - np.sin(angle(q, j1))) +
            abs(np.sin(angle(p, i2)) - np.sin(angle(q, j2))) +
            abs(np.sin(angle(p, i3)) - np.sin(angle(q, j3)))) / 3


def oposite_side(p, i):
    return np.linalg.norm(np.subtract(p[(i + 1) % 3], p[(i + 2) % 3]))


def dist_ratios(p, q, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    R1 = oposite_side(p, i1) / oposite_side(q, j1)
    R2 = oposite_side(p, i2) / oposite_side(q, j2)
    R3 = oposite_side(p, i3) / oposite_side(q, j3)
    return np.std([R1, R2, R3])


def dist_desc(dp, dq, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return (np.linalg.norm(np.subtract(dp[i1], dq[j1])) +
            np.linalg.norm(np.subtract(dp[i2], dq[j2])) +
            np.linalg.norm(np.subtract(dp[i3], dq[j3]))) / 3


def distance(triang1, triang2):
    p = triang1[:6]
    p = [np.array([p[i], p[j]]) for i, j in TRIANG_TO_P]
    dp = triang1[6:]
    q = triang2[:6]
    q = [np.array([q[i], q[j]]) for i, j in TRIANG_TO_P]
    dq = triang2[6:]
    min_dist = float('inf')
    for idx in PERMS:
        dist_vec = [
            dist_angles(p, q, *idx),
            dist_ratios(p, q, *idx),
            dist_desc(dp, dq, *idx)
        ]
        dist = np.linalg.norm(dist_vec)
        min_dist = dist if dist < min_dist else min_dist
    return min_dist


def asmatrix(keypoints, descriptors, triangulation):
    num_triangles = triangulation.shape[0]
    desc_dim = len(descriptors[0])
    triangles = np.zeros((num_triangles, 6 + 3 * desc_dim))
    for i in range(num_triangles):
        idx = triangulation[i, :]
        kp = [keypoints[i].pt for i in idx]
        desc = [descriptors[i] for i in idx]
        triangles[i, :] = np.concatenate(kp + desc)
    return triangles
