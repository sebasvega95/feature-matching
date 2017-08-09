import cv2
import numpy as np
from feature_matching import TBMatcher, get_features


def test_tbmatch(ref_img, query_img, num_kp=20, min_match=10):
    '''
    Tests the triangle-based matcher by finding an homography and counting the
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

    tm = TBMatcher()
    matches = tm.match(ref_kp, query_kp, ref_desc, query_desc)
    src_pts, dst_pts = tm.get_points()

    if len(matches) >= min_match:
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()
    else:
        raise ValueError('Not enough matches were found')
    return np.sum(mask) / len(matches)
