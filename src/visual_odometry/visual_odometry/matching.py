# Import modules
import cv2


class Matcher:
    def __init__(self,
                 criterion = cv2.NORM_HAMMING
                 ):
        
        # Attributes
        self.matcher = cv2.BFMatcher(criterion, crossCheck=True)

    # Match features
    def match(self,
              descriptors_0,
              descriptor_1
              ):
        
        matches = self.matcher.match(descriptors_0, descriptor_1)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches
    