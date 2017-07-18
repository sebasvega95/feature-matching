import cv2
import matplotlib.tri as tri


def get_features(img, hessian_threshold=400, num_keypoints=None):
    '''
    Return keypoints and descriptors of SURF detection algorithm. For
    details see
    http://docs.opencv.org/3.1.0/df/dd2/tutorial_py_surf_intro.html.

    Parameters
    ----------
    img: ndarray
        Input image, should be single-channel.
    hessian_threshold:
        Hessian threshold used by SURF.
    num_keypoints:
        Number of desired keypoints to return.

    Returns
    -------
    keypoints: list
        Keypoints detected.
    descriptors: list
        Corresponding descriptors to the keypoints found.
    '''
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    keypoints, descriptors = surf.detectAndCompute(img, None)
    return keypoints[:num_keypoints], descriptors[:num_keypoints]


def get_triangulation(keypoints):
    '''
    Calculate Delaunay triangulation of given SURF keypoints.

    Parameters
    ----------
    keypoints: list
        Keypoints detected by SURF.

    Returns
    -------
    triangulation: ndarray
        Delaunay triangulation of keypoints as a matrix, where each row
        denotes a triangle and each column the indices in the keypoints
        list that form said triangle.
    '''
    x, y = zip(*[k.pt for k in keypoints])
    triangulation = tri.Triangulation(x, y)
    return triangulation.get_masked_triangles()
