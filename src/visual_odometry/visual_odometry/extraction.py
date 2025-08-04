# Import modules
import numpy as np
import cv2


class Extractor:
    def __init__(self):
        pass

    # Detect ORB keypoints and compute descriptors
    def extractORB(self,
                   frame: np.ndarray
                   ):
        self.orb = cv2.ORB_create()
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    # Detect Shi-Tomasi stable keypoints
    def extractShiTomasi(self,
                         frame: np.ndarray
                         ):
        return cv2.goodFeaturesToTrack(frame, maxCorners=500, qualityLevel=0.01, minDistance=7)
