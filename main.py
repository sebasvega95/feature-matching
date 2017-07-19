#!env/bin/python
import cv2
import matplotlib.pyplot as plt
import triang2vec
from draw import draw_triangulation
from features import get_features, get_triangulation

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

    knn_train = triang2vec.asmatrix(ref_kp, ref_desc, ref_triangulation)
    knn_test = triang2vec.asmatrix(query_kp, query_desc, query_triangulation)

    print('Plotting')
    img1 = draw_triangulation(ref_img, ref_kp, ref_triangulation)
    img2 = draw_triangulation(query_img, query_kp, query_triangulation)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()
