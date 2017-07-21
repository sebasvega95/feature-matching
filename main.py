#!env/bin/python
import cv2
import matplotlib.pyplot as plt
import triang2vec
from datetime import timedelta
from draw import draw_triangulation
from features import get_features, get_triangulation
from sklearn.neighbors import NearestNeighbors
from time import time

if __name__ == '__main__':
    print('Opening images')
    ref_img = cv2.imread('book_covers/Reference/001.jpg')
    query_img = cv2.imread('book_covers/Droid/001.jpg')
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    print('Extracting features')
    ref_kp, ref_desc = get_features(ref_gray, num_keypoints=500)
    query_kp, query_desc = get_features(query_gray, num_keypoints=500)

    print('Calculating Triangulation')
    ref_triangulation = get_triangulation(ref_kp)
    query_triangulation = get_triangulation(query_kp)

    print('Training KNN')
    knn_train = triang2vec.asmatrix(ref_kp, ref_desc, ref_triangulation)
    knn_test = triang2vec.asmatrix(query_kp, query_desc, query_triangulation)

    neigh = NearestNeighbors(n_neighbors=2, metric=triang2vec.distance)
    neigh.fit(knn_train)

    print('Testing KNN')
    start = time()
    distances, indices = neigh.kneighbors(knn_test)
    end = time()
    print('Testing done in', timedelta(seconds=end - start))

    print('Lowe\'s ratio test')
    start = time()
    good = []
    num_matches = indices.shape[0]
    for i in range(num_matches):
        m_dist, n_dist = distances[i]
        if m_dist < 0.7 * n_dist:
            good.append([i, indices[i, 0]])
    # print(good)
    end = time()
    print('Final matching done in', timedelta(seconds=end - start))
    print('\tNum. matches:', len(good))
