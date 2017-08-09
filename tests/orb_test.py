import cv2
import numpy as np

FLANN_INDEX_LSH = 6


def test_orbmatch(ref_img, query_img, num_kp=None, min_match=10):
    '''
    Tests the ORB matcher by finding an homography and counting the
    percentage of inliers in said homography.

    Parameters
    ----------
    ref_img: ndarray
        Reference image, must have just one channel (greyscale).
    query_img: ndarray
        Query image, must have just one channel (greyscale).
    num_kp: int or None
        Number of keypoints to extract, None if all that are found.
    min_match: int
        Minimum number of point matches to be considered as valid for
        constructing the homography.

    Returns
    -------
    inliers: float
        The ratio between the number of point matches inside the homography and
        the total number of matches.
    num_kp_found: tuple (int, int)
        Number of keypoints found in both images.
    '''
    orb = cv2.ORB_create()

    ref_kp, ref_desc = orb.detectAndCompute(ref_img, None)
    query_kp, query_desc = orb.detectAndCompute(query_img, None)
    ref_kp = ref_kp[:num_kp]
    ref_desc = ref_desc[:num_kp]
    query_kp = query_kp[:num_kp]
    query_desc = query_desc[:num_kp]
    num_kp_found = (len(ref_kp), len(query_kp))

    index_params = {
        'algorithm': FLANN_INDEX_LSH,
        'table_number': 12,
        'key_size': 20,
        'multi_probe_level': 2
    }
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ref_desc, query_desc, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= min_match:
        src_pts = np.array([ref_kp[m.queryIdx].pt for m in good])
        dst_pts = np.array([query_kp[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()
    else:
        raise ValueError('Not enough matches were found')
    inliers = np.sum(mask) / len(good)
    return inliers, num_kp_found
