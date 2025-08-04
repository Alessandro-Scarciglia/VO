# Import modules
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


class Estimator:
    def __init__(self,
                 k_mtx: np.ndarray
                 ):
        
        # Attributes
        self.k_mtx = k_mtx

    @staticmethod
    def rotation_geodesic_distance(R1, R2):
        
        # Compute angular distance between R_t and R_t+1
        R = R1.T @ R2
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return theta
    
    @staticmethod
    def choose_best_RT(U, Vt, t_prev, R_prev):

        # Canonic rotational matrix
        W = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
        
        # Four possible solutions
        rotm_1 = U @ W @ Vt
        rotm_2 = U @ W.T @ Vt
        t = U[:, 2]

        # Select R by geodetic distance
        geo_dist_1 = Estimator.rotation_geodesic_distance(rotm_1, R_prev)
        geo_dist_2 = Estimator.rotation_geodesic_distance(rotm_2, R_prev)
        
        if geo_dist_1 < geo_dist_2:
            rotm = rotm_1
        else:
            rotm = rotm_2

        # Select compute direction of motion
        dir_prev = t_prev / np.linalg.norm(t_prev)
        dir_act = t / np.linalg.norm(t)

        # Compute similarity score with t and -t
        sim_score_pos = np.dot(dir_act, dir_prev)
        sim_score_neg = np.dot(-dir_act, dir_prev)

        # Choose the most similar
        if sim_score_pos < sim_score_neg:
            t *= -1

        return rotm, t


    # Compute relative motion from DMatch objects
    def compute_relative_motion(self,
                                keypoints_0,
                                keypoints_1,
                                matches,
                                prior_pose,
                                criterion=cv2.FM_RANSAC):
        
        # Estrai i punti corrispondenti da match e keypoints
        points_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matches])
        points_1 = np.float32([keypoints_1[m.trainIdx].pt for m in matches])

        # Calcola la matrice essenziale filtrando gli outliers
        e_mtx, _ = cv2.findEssentialMat(points_0, points_1, self.k_mtx, method=criterion, prob=0.99, threshold=1.)

        # Decomponi per ottenere R|t
        U, _, Vt = np.linalg.svd(e_mtx)

        # Enforce determinant of U and Vt
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1

        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1

        # Choose correct R|t given the previous state
        rotm_01, t_01 = Estimator.choose_best_RT(U, Vt, prior_pose[:3, 3], prior_pose[:3, :3])

        # Build the homogeneous matrix
        h_01 = np.eye(4)
        h_01[:3, :3] = rotm_01
        h_01[:3, 3] = t_01.ravel() 

        return h_01