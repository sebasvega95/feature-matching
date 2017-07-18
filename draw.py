import cv2
import numpy as np


def draw_triangulation(img, keypoints, triangulation):
    '''
    Draw the Delaunay triangulation of the SURF keypoints of an image.

    Parameters
    ----------
    img: ndarray
        Input image.
    keypoints: list
        Keypoints detected by SURF.
    triangulation: ndarray
        Delaunay triangulation of keypoints as a matrix, where each row
        denotes a triangle and each column the indices in the keypoints
        list that form said triangle.

    Returns
    -------
    out_img: ndarray
        Same image as img, but with the triangulation drawn over it.
    '''
    out_img = np.copy(img)
    for t in triangulation:
        p = [keypoints[ti].pt for ti in t]
        p1, p2, p3 = [(int(x), int(y)) for x, y in p]
        cv2.line(out_img, p1, p2, color=(255, 0, 0), thickness=1)
        cv2.line(out_img, p1, p3, color=(255, 0, 0), thickness=1)
        cv2.line(out_img, p2, p3, color=(255, 0, 0), thickness=1)
    return out_img
