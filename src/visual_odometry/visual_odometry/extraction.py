# Import modules
import numpy as np
import cv2


class Extractor:
    def __init__(self):
        
        # Attributes
        self.orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    def extract_features(self,
                        frame: np.ndarray
                        ):
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
