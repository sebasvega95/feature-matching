#!env/bin/python
import cv2
import matplotlib.pyplot as plt
import triang2vec
from datetime import timedelta
from draw import draw_triangulation
from features import get_features, get_triangulation
from kpmap import KeyPointMapper
from sklearn.neighbors import NearestNeighbors
from time import time

if __name__ == '__main__':
    print('Opening images')
    ref_img = cv2.imread('book_covers/Reference/001.jpg')
    query_img = cv2.imread('book_covers/Droid/001.jpg')
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    print('Extracting features')
    ref_kp, ref_desc = get_features(ref_gray, num_keypoints=20)
    query_kp, query_desc = get_features(query_gray, num_keypoints=20)
    ref_mapper = KeyPointMapper(ref_kp)
    query_mapper = KeyPointMapper(query_kp)

    print('Calculating Triangulation')
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

    # print('Lowe\'s ratio test')
    # good = []
    # num_matches = indices.shape[0]
    # print('Num. matches before:', num_matches)
    # ratio = 0.8
    # for i in range(num_matches):
    #     m_dist, n_dist = distances[i, :]
    #     if m_dist < ratio * n_dist:
    #         good.append([i, indices[i, 0]])
    # print('Num. matches after: ', len(good))

    print('Point matching')
    start = time()
    matches_set = set()
    matches = []
    print(len(ref_kp), len(query_kp))
    for query_idx, (ref_idx, dist) in enumerate(zip(indices, distances)):
        ref_idx = ref_idx[0]
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
                # matches.append(
                #     cv2.DMatch(
                #         _queryIdx=query_pt_idx,
                #         _trainIdx=ref_pt_idx,
                #         _distance=dist))
    end = time()
    print('\tMatching done in', timedelta(seconds=end - start))
    img_matches = cv2.drawMatches(
        ref_img, ref_kp, query_img, query_kp, matches, None, flags=2)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.show()
