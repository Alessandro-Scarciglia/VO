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

    # Compute relative motion from DMatch objects
    def compute_relative_motion(self,
                                keypoints_0,
                                keypoints_1,
                                matches,
                                criterion=cv2.FM_RANSAC):
        
        # Estrai i punti corrispondenti da match e keypoints
        points_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matches])
        points_1 = np.float32([keypoints_1[m.trainIdx].pt for m in matches])

        # Calcola la matrice essenziale
        e_mtx, inliers = cv2.findEssentialMat(points_0, points_1, self.k_mtx,
                                              method=criterion, prob=0.99, threshold=2)
        
        #print(f"Inliers: {int(inliers.sum())} | Inlier Ratio: {int(inliers.sum())/len(points_0)}")

        # Decomponi per ottenere R|t
        _, rotm_01, t_01, _ = cv2.recoverPose(e_mtx, points_0, points_1, self.k_mtx, mask=inliers)

        # Matrice omogenea 4x4
        h_01 = np.eye(4)
        h_01[:3, :3] = rotm_01
        h_01[:3, 3] = t_01.ravel()
        h_01[:3, 3] /= np.linalg.norm(h_01[:3, 3])

        # r = R.from_matrix(rotm_01)
        # angles = r.as_euler('xyz', degrees=True)  # oppure degrees=False per rad
        # theta_x = angles[0]
        # print(theta_x)

        return h_01