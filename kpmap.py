class KeyPointMapper:
    '''
    Maps a list of keypoint coordinates to their indices in an image.
    '''

    def __init__(self, keypoints):
        '''
        Stores the necessary data for indexing descriptors and indices by
        keypoints.

        Parameters
        ----------
        keypoints: list of KeyPoint
            Keypoints detected by SURF.
        '''
        self._mapper = {}
        for i, kp in enumerate(keypoints):
            self._mapper[kp.pt] = i

    def search(self, keypoint):
        '''
        Returns the index in the image and the descriptor corresponding to a
        keypoint

        Parameters
        ----------
        keypoint: tuple (int, int)
            Keypoint to search.
        Returns
        -------
        index: int
            Index of the keypoint in the image, or -1 if it's not found.
        '''
        try:
            index = self._mapper[keypoint]
        except KeyError:
            index = -1
        return index
