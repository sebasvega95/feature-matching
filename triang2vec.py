import numpy as np

PERMS = [[(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 2), (2, 1)],
         [(0, 1), (1, 0), (2, 2)], [(0, 1), (1, 2), (2, 0)],
         [(0, 2), (1, 0), (2, 1)], [(0, 2), (1, 1), (2, 0)]]

TRIANG_TO_POINT_IDX = [(0, 1), (2, 3), (4, 5)]

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


class KeyPointMapper:
    '''
    Represents two triangulations of keypoints in a convenient way to calculate
    the proposed distance between triangles.
    '''

    def __init__(self, keypoints, descriptors):
        '''
        Stores the necessary data for indexing descriptors and indices by
        keypoints.

        Parameters
        ----------
        keypoints: list of KeyPoint
            Keypoints detected by SURF.
        descriptors: ndarray
            Corresponding descriptors to the keypoints.
        '''
        self._mapper = {}
        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            self._mapper[kp.pt] = {}
            self._mapper[kp.pt]['descriptor'] = desc
            self._mapper[kp.pt]['index'] = i

    def search(self, keypoint):
        '''
        Returns the index in the image and the descriptor corresponding to a
        keypoint

        Parameters
        ----------
        keypoint: tuple (int, int)
            Keypoint to search.
        Returns
        -------
        index: int
            Index of the keypoint in the image.
        descriptor: ndarray
            Corresponding descriptors to the keypoint.
        '''
        index = self._mapper[kp.pt]['index']
        descriptor = self._mapper[kp.pt]['descriptor']
        return index, descriptor


class TriangleDistance:
    '''
    Class to that contains the necessary data and methods for calculating the
    proposed distance between two triangles.
    '''

    def __init__(self, mapper1, mapper2):
        '''
        Stores the necessary data for calculating distances.

        Parameters
        ----------
        mapper#: KeyPointMapper
            Maps keypoints to descriptors and indices in an image.
        '''
        self.mapper = (mapper1, mapper2)

    def __call__(self, triang1, triang2):
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
        p, dp = self._unroll(triang1, 0)
        q, dq = self._unroll(triang2, 1)
        dist = float('inf')
        for idx in PERMS:
            dist_vec = [
                self._dist_angles(p, q, *idx),
                self._dist_ratios(p, q, *idx),
                self._dist_desc(dp, dq, *idx)
            ]
            _dist = np.linalg.norm(dist_vec)
            dist = _dist if _dist < dist else dist
        return dist

    def _unroll(self, triangle, which):
        '''
        Takes the indices of the points that make up a triangle and returns the
        corresponding keypoints and descriptors.

        Parameters
        ----------
        triangle: ndarray
            Triangle as the indices of the points that make it.
        which: int
            0 if the triangle is from the first image, 1 otherwise

        Returns
        -------
        points: list of ndarray
            A list of the (x, y) coordinates of the points.
        descriptors: list of ndarray
            A list of the SURF descriptors of the points.
        '''
        idx = triangle.tolist()
        print(idx)
        points = [triangle[[i, j]] for i, j in TRIANG_TO_POINT_IDX]
        descriptors = [
            self._mapper[which].search(triangle[i], triangle[j])
            for p in TRIANG_TO_POINT_IDX
        ]
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

    def _dist_angles(p, q, idx1, idx2, idx3):
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

    def _dist_ratios(p, q, idx1, idx2, idx3):
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
        R1 = self._oposite_side(p, i1) / self._oposite_side(q, j1)
        R2 = self._oposite_side(p, i2) / self._oposite_side(q, j2)
        R3 = self._oposite_side(p, i3) / self._oposite_side(q, j3)
        return np.std([R1, R2, R3])

    def _dist_desc(dp, dq, idx1, idx2, idx3):
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
        return (np.linalg.norm(dp[i1] - dq[j1]) + np.linalg.norm(
            dp[i2] - dq[j2]) + np.linalg.norm(dp[i3] - dq[j3])) / 3
