import numpy as np

PERMS = [[(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 2), (2, 1)],
         [(0, 1), (1, 0), (2, 2)], [(0, 1), (1, 2), (2, 0)],
         [(0, 2), (1, 0), (2, 1)], [(0, 2), (1, 1), (2, 0)]]

# TRIANG_TO_POINT_IDX = [(0, 1), (2, 3), (4, 5)]

# def unroll(triang):
#     point = triang[:6]
#     point = [np.array([point[i], point[j]]) for i, j in TRIANG_TO_POINT_IDX]
#     descriptor = triang[6:]
#     return point, descriptor

# def asmatrix(keypoints, descriptors, triangulation):
#     num_triangles = triangulation.shape[0]
#     desc_dim = len(descriptors[0])
#     triangles = np.zeros((num_triangles, 6 + 3 * desc_dim))
#     for i in range(num_triangles):
#         idx = triangulation[i, :]
#         kp = [keypoints[i].pt for i in idx]
#         desc = [descriptors[i] for i in idx]
#         triangles[i, :] = np.concatenate(kp + desc)
#     return triangles


class TriangulationMatrix:
    '''
    Represents a triangulation of a set of keypoints in a convenient way to
    calculate the proposed distance between triangles.
    '''

    def __init__(self, keypoints, descriptors, triangulation):
        '''
        Stores the necessary data for calculating distances.

        Parameters
        ----------
        keypoints: list
            Keypoints detected by SURF.
        descriptors: list
            Corresponding descriptors to the keypoints found.
        triangulation: ndarray
            Delaunay triangulation of keypoints as a matrix, where each row
            denotes a triangle and each column the indices in the keypoints
            list that form said triangle.
        '''
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.triangulation = triangulation

    def distance(self, triang1, triang2):
        '''
        Calculates the proposed distance between two triangles. This distance is
        the L2-norm of a vector whose components are the three distances
        described in the methods _dist_angles, _dist_ratios and _dist_desc. Note
        that there are six different ways of matching-up the points in the
        triangles when comparing them, so we take the configuration that has the
        smallest distance.

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
        p, dp = _unroll(triang1)
        q, dq = _unroll(triang2)
        dist = float('inf')
        for idx in PERMS:
            dist_vec = [
                _dist_angles(p, q, *idx),
                _dist_ratios(p, q, *idx),
                _dist_desc(dp, dq, *idx)
            ]
            _dist = np.linalg.norm(dist_vec)
            dist = _dist if _dist < dist else dist
        return dist

    def _unroll(self, triangle):
        '''
        Takes the indices of the points that make up a triangle and returns the
        corresponding keypoints and descriptors.

        Parameters
        ----------
        triangle: ndarray
            Triangle as the indices of the points that make it.

        Returns
        -------
        points: list
            A list of the (x, y) coordinates of the points.
        descriptors: list
            A list of the SURF descriptors of the points.
        '''
        points = [np.array(self.keypoints[i].pt) for i in triangle]
        descriptors = [np.array(self.descriptors[i]) for i in triangle]
        return points, descriptors

    def _angle(p, i):
        '''
        Angle corresponding to a point in a triangle.

        Parameters
        ----------
        p: list
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

    def _dist_angles(p, q, idx1, idx2, idx3):
        '''
        The angle distance between two triangles, computed as the arithmetic
        mean of the absolute value of the difference between the sines of the
        angles.

        Parameters
        ----------
        p: list
            A list of the (x, y) coordinates of the points of a triangle.
        q: list
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
        p: list
            A list of the (x, y) coordinates of the points.
        i: int
            Index of the point whose angle is to be returned.

        Returns
        -------
        length: float
            Length of the side oposite to p[i] in the triangle p.
        '''
        return np.linalg.norm(p[(i + 1) % 3] - p[(i + 2) % 3])

    def _dist_ratios(p, q, idx1, idx2, idx3):
        '''
        The ratio distance between two triangles, computed as the standard
        deviation of the ratios of the sides of the triangles.

        Parameters
        ----------
        p: list
            A list of the (x, y) coordinates of the points of a triangle.
        q: list
            A list of the (x, y) coordinates of the points of a triangle.
        idx#: tuple
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

    def _dist_desc(dp, dq, idx1, idx2, idx3):
        '''
        The descriptor distance between two triangles, computed as the
        arithmetic mean of the euclidean distance between the descriptors.

        Parameters
        ----------
        dp: list
            A list of the SURF descriptors of the points of a triangle.
        dq: list
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
        return (np.linalg.norm(dp[i1] - dq[j1]) + np.linalg.norm(
            dp[i2] - dq[j2]) + np.linalg.norm(dp[i3] - dq[j3])) / 3
