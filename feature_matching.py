import cv2
from tests import test_tbmatch, test_surfmatch

if __name__ == '__main__':
    ref_img = cv2.imread('test-images/monster1s.JPG', 0)
    query_img = cv2.imread('test-images/monster1s.rot.JPG', 0)

    tb_inliners, tb_num_kp = test_tbmatch(ref_img, query_img)
    surf_inliers, surf_num_kp = test_surfmatch(ref_img, query_img)

    print('Homography test')
    print('---------------')
    print('Num. points (ref, query)')
    print('Triangle-based:', tb_num_kp)
    print('SURF:          ', surf_num_kp)
    print('---------------')
    print('% inliers')
    print('Triangle-based: {:.3f}%'.format(tb_inliners * 100))
    print('SURF:           {:.3f}%'.format(surf_inliers * 100))
