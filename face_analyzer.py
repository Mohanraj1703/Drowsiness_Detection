import cv2
import numpy as np
from scipy.spatial import distance
from typing import Tuple


class FaceAnalyzer:
    """
    Stateless utility class for all facial metric calculations.
    Contains pure functions for EAR, MAR, Head Pose, and Gaze Ratio.
    No side effects — each method takes raw data in and returns a result.
    """

    @staticmethod
    def calculate_ear(eye_landmarks: np.ndarray) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR).
        A low EAR value indicates the eye is closing or closed.
        Formula: (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        """
        a = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        b = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        c = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (a + b) / (2.0 * c)

    @staticmethod
    def calculate_mar(mouth_landmarks: np.ndarray) -> float:
        """
        Calculates the Mouth Aspect Ratio (MAR).
        A high MAR value indicates the mouth is open (yawning).
        Formula: vertical_distance / horizontal_distance
        """
        a = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])  # Top-bottom
        b = distance.euclidean(mouth_landmarks[2], mouth_landmarks[3])  # Left-right
        return a / b if b != 0 else 0.0

    @staticmethod
    def calculate_head_pose(landmarks, w: int, h: int) -> Tuple[float, float]:
        """
        Estimates head Pitch (up/down) and Yaw (left/right) using OpenCV solvePnP.
        Maps 6 key facial landmarks to a generic 3D skull model.
        Returns: (pitch, yaw) in degrees.
        """
        face_2d = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in [1, 33, 263, 61, 291, 152]  # Nose, Eyes, Mouth corners, Chin
        ], dtype=np.float64)

        # Generic 3D model points in World Coordinates
        face_3d = np.array([
            [0.0,    0.0,    0.0],           # Nose tip
            [-165.0, -170.0, -135.0],        # Left eye outer corner
            [165.0,  -170.0, -135.0],        # Right eye outer corner
            [-150.0, 150.0,  -125.0],        # Left mouth corner
            [150.0,  150.0,  -125.0],        # Right mouth corner
            [0.0,    330.0,  -65.0]          # Chin
        ], dtype=np.float64)

        focal_length = 1 * w
        camera_mat = np.array(
            [[focal_length, 0, w / 2],
             [0, focal_length, h / 2],
             [0, 0, 1]],
            dtype=np.float64
        )
        dist_coef = np.zeros((4, 1), dtype=np.float64)

        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, camera_mat, dist_coef)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return angles[0], angles[1]  # pitch, yaw

    @staticmethod
    def calculate_gaze_ratio(landmarks, w: int, h: int) -> float:
        """
        Calculates the average gaze ratio based on iris position within the eye.
        A ratio near 0.5 = looking forward. Near 0 = left. Near 1 = right.
        Requires MediaPipe iris landmarks (478 total points).
        """
        if len(landmarks) < 478:
            return 0.5  # Default to forward-looking if iris data unavailable

        def _iris_ratio(left_idx: int, right_idx: int, iris_idx: int) -> float:
            p_left = np.array([landmarks[left_idx].x * w,  landmarks[left_idx].y * h])
            p_right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])
            p_iris = np.array([landmarks[iris_idx].x * w,  landmarks[iris_idx].y * h])
            width = distance.euclidean(p_left, p_right)
            dist_to_left = distance.euclidean(p_left, p_iris)
            return dist_to_left / width if width > 0 else 0.5

        left_ratio = _iris_ratio(33, 133, 468)
        right_ratio = _iris_ratio(362, 263, 473)
        return (left_ratio + right_ratio) / 2.0
