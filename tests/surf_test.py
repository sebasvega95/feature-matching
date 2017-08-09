import cv2
import numpy as np
from feature_matching import get_features

FLANN_INDEX_KDTREE = 0


def test_surfmatch(ref_img, query_img, num_kp=20, min_match=10):
    '''
    Tests the SURF matcher by finding an homography and counting the
    percentage of inliers in said homography.

    Parameters
    ----------
    ref_img: ndarray
        Reference image, must have just one channel (greyscale).
    query_img: ndarray
        Query image, must have just one channel (greyscale).
    num_kp: int
        Number of keypoints to extract
    min_match: int
        Minimum number of point matches to be considered as valid for
        constructing the homography.

    Returns
    -------
    inliers: float
        The ratio between the number of point matches inside the homography and
        the total number of matches.
    '''
    ref_kp, ref_desc = get_features(ref_img, num_keypoints=num_kp)
    query_kp, query_desc = get_features(query_img, num_keypoints=num_kp)

    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ref_desc, query_desc, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= min_match:
        src_pts = np.array([ref_kp[m.queryIdx].pt for m in good])
        dst_pts = np.array([query_kp[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()
    else:
        raise ValueError('Not enough matches were found')
    return np.sum(mask) / len(good)
