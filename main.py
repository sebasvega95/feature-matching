import cv2
import matplotlib.pyplot as plt
import numpy as np
import triang2vec
from datetime import timedelta
from features import get_features, get_triangulation
from kpmap import KeyPointMapper
from sklearn.neighbors import NearestNeighbors
from time import time

DIST_THRESHOLD = 0.3
MIN_MATCH_COUNT = 10


def point_match(knn_train, knn_test, indices, distances, ref_mapper,
                query_mapper):
    '''
    Finds a point match given the triangle matches. This function aligns the
    points in the triangle so that the distance between the triangles matches
    the one in the distances vector.

    Parameters
    ----------
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

    Returns
    -------
    matches: list of DMatch
        Match between the points.
    src_pts: ndarray
        Coordinates of matched points in reference image.
    dst_pts: ndarray
        Coordinates of matched points in query image.
    '''
    matches_set = set()
    matches = []
    src_pts = []
    dst_pts = []
    for query_idx, (ref_idx, dist) in enumerate(zip(indices, distances)):
        if dist > DIST_THRESHOLD:
            continue
        ref_triang = knn_train[ref_idx, :]
        ref_points, _ = triang2vec.unroll(ref_triang)
        query_triang = knn_test[query_idx, :]
        query_points, _ = triang2vec.unroll(query_triang)
        alignment = triang2vec.best_point_alignment(ref_triang, query_triang)
        for i, j in alignment:
            ref_pt = tuple(ref_points[i])
            query_pt = tuple(query_points[j])
            ref_pt_idx = ref_mapper.search(ref_pt)
            query_pt_idx = query_mapper.search(query_pt)
            if (ref_pt_idx, query_pt_idx) not in matches_set:
                matches_set.add((ref_pt_idx, query_pt_idx))
                matches.append(cv2.DMatch(ref_pt_idx, query_pt_idx, dist))
                src_pts.append(ref_pt)
                dst_pts.append(query_pt)
    return matches, np.array(src_pts), np.array(dst_pts)


if __name__ == '__main__':
    print('Opening images')
    ref_img = cv2.imread('test-images/monster1s.JPG')
    query_img = cv2.imread('test-images/monster1s.rot.JPG')
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    print('Extracting features')
    num_kp = 20
    print('\tNum. keypoints:', num_kp)
    ref_kp, ref_desc = get_features(ref_gray, num_keypoints=num_kp)
    query_kp, query_desc = get_features(query_gray, num_keypoints=num_kp)
    ref_mapper = KeyPointMapper(ref_kp)
    query_mapper = KeyPointMapper(query_kp)

    print('Calculating triangulation')
    ref_triangulation = get_triangulation(ref_kp)
    query_triangulation = get_triangulation(query_kp)

    print('Training KNN')
    knn_train = triang2vec.asmatrix(ref_kp, ref_desc, ref_triangulation)
    knn_test = triang2vec.asmatrix(query_kp, query_desc, query_triangulation)

    neigh = NearestNeighbors(n_neighbors=1, metric=triang2vec.distance)
    neigh.fit(knn_train)

    print('Triangle matching')
    start = time()
    distances, indices = neigh.kneighbors(knn_test)
    end = time()
    print('\tMatching done in', timedelta(seconds=end - start))

    distances = distances.ravel()
    indices = indices.ravel()

    print('Point matching')
    matches, src_pts, dst_pts = point_match(
        knn_train, knn_test, indices, distances, ref_mapper, query_mapper)
    print('\tNum. matches:', len(matches))

    if len(matches) >= MIN_MATCH_COUNT:
        print('Finding homography')
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()
        print('\tInliers: {:.3f}%'.format(np.sum(mask) / len(matches) * 100))
        mask = mask.tolist()
    else:
        print('Not enough matches were found')
        mask = None

    img_matches = cv2.drawMatches(
        ref_img,
        ref_kp,
        query_img,
        query_kp,
        matches,
        None,
        flags=2,
        matchesMask=mask)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.show()
