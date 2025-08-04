# Import modules
import numpy as np
import cv2
import json
from tqdm import tqdm
from display import plot_trajectory, draw_matches
from utils import calculate_intrinsic_matrix

# Custom modules
from preprocessing import Preprocessor
from extraction import Extractor
from matching import Matcher
from motion_estimation import Estimator
from motion_integration import Integrator


class VisualOdometry:
    def __init__(self,
                 fov: float,
                 resolution: tuple,
                 sigma: float = 0.5,
                 initial_pose: np.ndarray = None
                 ):

        # Attributes
        self.initial_pose = initial_pose
        self.trajectory = [initial_pose]
        k_mtx = calculate_intrinsic_matrix(fov=fov, resolution=resolution)
        self.last_rel = initial_pose
        self.last_rel[1:3, 3] += 1e-3

        # Agents
        self.preprocessor = Preprocessor(sigma=sigma)
        self.extractor = Extractor()
        self.matcher = Matcher()
        self.motion_estimator = Estimator(k_mtx=k_mtx)
        self.integrator = Integrator()

    # Visual Odometry 
    def update_trajectory(self,
                          frame_0: np.ndarray,
                          frame_1: np.ndarray
                          ):
        
        # Preprocess the frames
        frame_0_prep = self.preprocessor.preprocess(frame_0)
        frame_1_prep = self.preprocessor.preprocess(frame_1)

        # Extract features
        keypoints_0, descriptors_0 = self.extractor.extractORB(frame_0_prep)
        keypoints_1, descriptors_1 = self.extractor.extractORB(frame_1_prep)

        # Match features
        matches = self.matcher.match(descriptors_0, descriptors_1)

        # Compute relative motion from 0 to 1
        pose_0_to_1 = self.motion_estimator.compute_relative_motion(keypoints_0, keypoints_1, matches, self.last_rel)

        # Integrate motion with respect to the trajectory
        rotm_1 = pose_0_to_1[:3, :3] @ self.trajectory[-1][:3, :3]
        t_1 = self.trajectory[-1][:3, 3] + self.trajectory[-1][:3, :3] @ pose_0_to_1[:3, 3]

        # Build homogeneous matrix
        pose_1 = np.eye(4)
        pose_1[:3, :3] = rotm_1
        pose_1[:3, 3] = t_1.ravel() 
        self.trajectory.append(pose_1)
        self.last_rel = pose_0_to_1

        return pose_1



if __name__ == "__main__":

    # Set resolution
    RES = 1024
    SKIP = 10

    # Load data
    with open(f"/home/visione/Projects/VO/src/dataset/Orbit_{RES}/transforms.json", "r") as train_fopen:
        df = json.load(train_fopen)
        fov = df["camera_angle_x"]

    # Pose initialization
    initial_pose = np.eye(4)

    # Visual Odometry class
    vo = VisualOdometry(initial_pose=initial_pose,
                        sigma=1.,
                        fov=fov,
                        resolution=(RES, RES))

    for i in tqdm(range(0, 360-SKIP, SKIP)):

        # Two example frames
        frame_0 = cv2.imread(f"/home/visione/Projects/VO/src/dataset/Orbit_{RES}/" + str(i).zfill(3) + ".png", 0)
        frame_1 = cv2.imread(f"/home/visione/Projects/VO/src/dataset/Orbit_{RES}/" + str(i+SKIP).zfill(3) + ".png", 0)

        if frame_1 is None:
            continue

        vo.update_trajectory(frame_0, frame_1)
    
    # Display trajetory 
    plot_trajectory(vo.trajectory, axis_length=0.5)
