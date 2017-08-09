import cv2
import matplotlib.pyplot as plt
import numpy as np
from feature_matching import TBMatcher, get_features

MIN_MATCH_COUNT = 10

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

    tm = TBMatcher()
    matches = tm.match(ref_kp, query_kp, ref_desc, query_desc)
    src_pts, dst_pts = tm.get_points()

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
