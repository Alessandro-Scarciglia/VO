# Import modules
import numpy as np
import cv2


class Preprocessor:
    def __init__(self,
                 sigma: float = 0.5
                 ):
        
        # Attributes
        self.sigma = sigma
        self.kernel_size = 2 * int(np.ceil(3 * self.sigma)) + 1
    
    def preprocess(self,
                   frame: np.ndarray
                   ):
        return cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), self.sigma)
