# Import modules
import numpy as np
import cv2

# Custom modules
from preprocessing import Preprocessor
from extraction import Extractor
from matching import Matcher
from motion_estimation import Estimator
from motion_integration import Integrator


class VisualOdometry:
    def __init__(self,
                 sigma: float = 0.5,
                 initial_pose: np.ndarray = None
                 ):

        # Agents
        self.preprocessor = Preprocessor(sigma=sigma)
        self.extractor = Extractor()
        self.matcher = Matcher()
        self.motion_estimator = Estimator()
        self.integrator = Integrator()

        # Attributes
        self.initial_pose = initial_pose
        self.trajectory = [initial_pose]

    # Visual Odometry 
    def update_trajectory(self,
                          frame_0: np.ndarray,
                          frame_1: np.ndarray
                          ):
        
        # Preprocess the frames
        frame_0_prep = self.preprocessor.preprocess(frame_0)
        frame_1_prep = self.preprocessor.preprocess(frame_1)

        # Extract features
        keypoints_0, descriptors_0 = self.extractor.extract_features(frame_0_prep)
        keypoints_1, descriptors_1 = self.extractor.extract_features(frame_1_prep)

        # Match features
        matches = self.matcher.match(descriptors_0, descriptors_1)

        matched_img = cv2.drawMatches(frame_0, keypoints_0,
                                      frame_1, keypoints_1,
                                      matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        cv2.imshow("", matched_img)
        cv2.waitKey(100)



if __name__ == "__main__":

    for i in range(0, 80):
        # Two example frames
        frame_0 = cv2.imread("/home/visione/Projects/VO/src/dataset/sat_light_360/" + str(i).zfill(3) + ".png")
        frame_1 = cv2.imread("/home/visione/Projects/VO/src/dataset/sat_light_360/" + str(i+3).zfill(3) + ".png")

        # Pose initialization
        initial_pose = np.eye(4)

        # Visual Odometry class
        vo = VisualOdometry(initial_pose=initial_pose)
        vo.update_trajectory(frame_0, frame_1)
