import numpy as np

DIM_DESC = 64
DIM_PT = 2
PERMS = [[(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 2), (2, 1)],
         [(0, 1), (1, 0), (2, 2)], [(0, 1), (1, 2), (2, 0)],
         [(0, 2), (1, 0), (2, 1)], [(0, 2), (1, 1), (2, 0)]]
TRIANG_DESC_IDX = [6, 70, 134]
TRIANG_POINT_IDX = [0, 2, 4]


def distance(triang1, triang2):
    '''
        Calculates the proposed distance between two triangles. This distance
        is the L2-norm of a vector whose components are the three distances
        described in the methods _dist_angles, _dist_ratios and _dist_desc.
        Note that there are six different ways of matching-up the points in the
        triangles when comparing them, so we take the configuration that has
        the smallest distance.

        Parameters
        ----------
        triang1: ndarray
            Triangle as the indices of the points that make it.
        triang2: ndarray
            Triangle as the indices of the points that make it.

        Returns
        -------
        dist: float
            Proposed distance between the triangles.
        '''
    p, dp = unroll(triang1)
    q, dq = unroll(triang2)
    dist = float('inf')
    for idx in PERMS:
        dist_vec = [
            dist_angles(p, q, *idx),
            dist_ratios(p, q, *idx),
            dist_desc(dp, dq, *idx)
        ]
        _dist = np.linalg.norm(dist_vec)
        dist = _dist if _dist < dist else dist
    return dist


def best_point_alignment(triang1, triang2):
    '''
    Calculates the same as triang2vec.distance, but returns the best matching
    between the triangles instead of the distance between them.

    Parameters
    ----------
    triang1: ndarray
        Triangle as the indices of the points that make it.
    triang2: ndarray
        Triangle as the indices of the points that make it.

    Returns
    -------
    best_match: list of tuples (int, int)
        Best aligment between the points of the triangles.
    '''
    p, dp = unroll(triang1)
    q, dq = unroll(triang2)
    dist = float('inf')
    for idx in PERMS:
        dist_vec = [
            dist_angles(p, q, *idx),
            dist_ratios(p, q, *idx),
            dist_desc(dp, dq, *idx)
        ]
        _dist = np.linalg.norm(dist_vec)
        if _dist < dist:
            dist = _dist
            best_match = idx
    return best_match


def unroll(triangle):
    '''
    Takes the indices of the points that make up a triangle and returns the
    corresponding keypoints and descriptors.

    Parameters
    ----------
    triangle: ndarray
        Triangle as the indices of the points that make it.

    Returns
    -------
    points: list of ndarray
        A list of the (x, y) coordinates of the points.
    descriptors: list of ndarray
        A list of the SURF descriptors of the points.
    '''
    points = [triangle[i:i + DIM_PT] for i in TRIANG_POINT_IDX]
    descriptors = [triangle[i:i + DIM_DESC] for i in TRIANG_DESC_IDX]
    return points, descriptors


def _angle(p, i):
    '''
    Angle corresponding to a point in a triangle.

    Parameters
    ----------
    p: list of ndarray
        A list of the (x, y) coordinates of the points.
    i: int
        Index of the point whose angle is to be returned.

    Returns
    -------
    angle: float
        The angle of point p[i] in the triangle p.
    '''
    a = p[i] - p[(i + 1) % 3]
    b = p[i] - p[(i + 2) % 3]
    return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def dist_angles(p, q, idx1, idx2, idx3):
    '''
    The angle distance between two triangles, computed as the arithmetic
    mean of the absolute value of the difference between the sines of the
    angles.

    Parameters
    ----------
    p: list of ndarray
        A list of the (x, y) coordinates of the points of a triangle.
    q: list of ndarray
        A list of the (x, y) coordinates of the points of a triangle.
    idx#: tuple
        Indices of the matched-up points between the triangles.

    Returns
    -------
    dist: float
        Proposed angle distance between the triangles.
    '''
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return (abs(np.sin(_angle(p, i1)) - np.sin(_angle(q, j1))) +
            abs(np.sin(_angle(p, i2)) - np.sin(_angle(q, j2))) +
            abs(np.sin(_angle(p, i3)) - np.sin(_angle(q, j3)))) / 3


def _oposite_side(p, i):
    '''
    Length of the side oposite to a point in a triangle.

    Parameters
    ----------
    p: list of ndarray
        A list of the (x, y) coordinates of the points.
    i: int
        Index of the point whose angle is to be returned.

    Returns
    -------
    length: float
        Length of the side oposite to p[i] in the triangle p.
    '''
    return np.linalg.norm(p[(i + 1) % 3] - p[(i + 2) % 3])


def dist_ratios(p, q, idx1, idx2, idx3):
    '''
    The ratio distance between two triangles, computed as the standard
    deviation of the ratios of the sides of the triangles.

    Parameters
    ----------
    p: list of ndarray
        A list of the (x, y) coordinates of the points of a triangle.
    q: list of ndarray
        A list of the (x, y) coordinates of the points of a triangle.
    idx#: tuple (int, int)
        Indices of the matched-up points between the triangles.

    Returns
    -------
    dist: float
        Proposed ratio distance between the triangles.
    '''
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    R1 = _oposite_side(p, i1) / _oposite_side(q, j1)
    R2 = _oposite_side(p, i2) / _oposite_side(q, j2)
    R3 = _oposite_side(p, i3) / _oposite_side(q, j3)
    return np.std([R1, R2, R3])


def dist_desc(dp, dq, idx1, idx2, idx3):
    '''
    The descriptor distance between two triangles, computed as the
    arithmetic mean of the euclidean distance between the descriptors.

    Parameters
    ----------
    dp: list of ndarray
        A list of the SURF descriptors of the points of a triangle.
    dq: list of ndarray
        A list of the SURF descriptors of the points of a triangle.
    idx#: tuple
        Indices of the matched-up points between the triangles.

    Returns
    -------
    dist: float
        Proposed descriptor distance between the triangles.
    '''
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return (np.linalg.norm(dp[i1] - dq[j1]) + np.linalg.norm(dp[i2] - dq[j2]) +
            np.linalg.norm(dp[i3] - dq[j3])) / 3


def asmatrix(keypoints, descriptors, triangulation):
    '''
    Represents two triangulations of keypoints in a convenient way to calculate
    the proposed distance between triangles.

    Parameters
    ----------
    keypoints: list of KeyPoint
        Keypoints detected by SURF.
    descriptors: ndarray
        Corresponding descriptors to the keypoints.
    triangulation: ndarray
        Delaunay triangulation of keypoints as a matrix, where each row
        denotes a triangle and each column the indices in the keypoints
        list that form said triangle.

    Returns
    -------
    triangles: ndarray
        Keypoints coordinates and descriptors as a matrix. Each row is a
        triangle. Columns are arranged as follows:
            pt1.pos pt2.pos pt3.pos pt1.desc pt2.desc pt3.desc
        Where pt# indicates the three points of a triangle. pos means the
        (x, y) coordinates of each point in the image. desc denotes the 64-dim
        descriptor vector. So the number of columns of the matrix is
            3 * 2 + 3 * 64 = 198
    '''
    num_triangles = triangulation.shape[0]
    if descriptors.shape[1] != DIM_DESC:
        raise ValueError('SURF descriptor should be {}-dim'.format(DIM_DESC))
    triangles = np.zeros((num_triangles, 3 * DIM_PT + 3 * DIM_DESC))
    for i in range(num_triangles):
        idx = triangulation[i, :]
        kp = [keypoints[i].pt for i in idx]
        desc = [descriptors[i] for i in idx]
        triangles[i, :] = np.concatenate(kp + desc)
    return triangles
