import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .features import get_triangulation
from .kpmap import KeyPointMapper
from .triang2vec import (best_point_alignment, triang_distance, triang_matrix,
                         unroll)


class TBMatcher:
    '''
    Triangle-Based Matcher. Point matched based on the proposed triangle
    distance.

    Attributes
    ----------
    dist_threshold: float
        Maximum distance between triangles to be considered as a match.
    knn_train, knn_test: ndarray
        Keypoints coordinates and descriptors as a matrix.
    indices:
        Indices of the nearest triangles in the train matrix. This is,
        indices[i] is the index of the triangle in knn_train closest to the
        i-th triangle in knn_test.
    distances: ndarray
        Distance between the triangles.
    ref_mapper, query_mapper: KeyPointMapper
        Mapping between the points and their indices in the images.
    src_pts: ndarray
        Coordinates of matched points in reference image.
    dst_pts: ndarray
        Coordinates of matched points in query image.
    '''

    def __init__(self, dist_threshold=0.2):
        '''
        Sets the dist_threshold if passed as input, and initializes the
        coordinates of the points to be matched as empty lists.
        '''
        self.dist_threshold = dist_threshold
        self.src_pts = []
        self.dst_pts = []

    def get_points(self):
        '''
        Get the corrdinates of the matched points in both the reference and the
        query image. Can be useful for finding an homographic warp between both
        images given the matches.

        Returns
        -------
        src_pts: ndarray
            Coordinates of matched points in reference image.
        dst_pts: ndarray
            Coordinates of matched points in query image.
        '''
        return self.src_pts, self.dst_pts

    def _point_match(self):
        '''
        Finds a point match given the triangle matches. This function aligns
        the points in the triangle so that the distance between the triangles
        matches the one in the distances vector.

        Returns
        -------
        matches: list of DMatch
            Match between the points.
        '''
        matches_set = set()
        matches = []
        src_pts = []
        dst_pts = []
        search_space = zip(self.indices, self.distances)
        for query_idx, (ref_idx, dist) in enumerate(search_space):
            if dist > self.dist_threshold:
                continue
            ref_triang = self.knn_train[ref_idx, :]
            ref_points, _ = unroll(ref_triang)
            query_triang = self.knn_test[query_idx, :]
            query_points, _ = unroll(query_triang)
            alignment = best_point_alignment(ref_triang, query_triang)
            for i, j in alignment:
                ref_pt_idx = self.ref_mapper.search(ref_points[i])
                query_pt_idx = self.query_mapper.search(query_points[j])
                if (ref_pt_idx, query_pt_idx) not in matches_set:
                    matches_set.add((ref_pt_idx, query_pt_idx))
                    matches.append(cv2.DMatch(ref_pt_idx, query_pt_idx, dist))
                    src_pts.append(ref_points[i])
                    dst_pts.append(query_points[j])
        self.src_pts = np.array(src_pts)
        self.dst_pts = np.array(dst_pts)
        return matches

    def match(self, ref_kp, query_kp, ref_desc, query_desc):
        '''
        Matches the keypoints of two images, given also their descriptors.

        Parameters
        ----------
        ref_kp: list of KeyPoint
            Keypoints detected by SURF in the reference image.
        query_kp: list of KeyPoint
            Keypoints detected by SURF in the query image.
        ref_desc: ndarray
            Corresponding descriptors to the keypoints found in the reference
            image.
        query_desc: ndarray
            Corresponding descriptors to the keypoints found in the query
            image.

        Returns
        -------
        matches: list of DMatch
            Match between the points keypoints.
        '''
        self.ref_mapper = KeyPointMapper(ref_kp)
        self.query_mapper = KeyPointMapper(query_kp)
        ref_triangulation = get_triangulation(ref_kp)
        query_triangulation = get_triangulation(query_kp)
        self.knn_train = triang_matrix(ref_kp, ref_desc, ref_triangulation)
        self.knn_test = triang_matrix(query_kp, query_desc,
                                      query_triangulation)
        neigh = NearestNeighbors(n_neighbors=1, metric=triang_distance)
        neigh.fit(self.knn_train)
        distances, indices = neigh.kneighbors(self.knn_test)
        self.distances = distances.ravel()
        self.indices = indices.ravel()
        matches = self._point_match()
        return matches
