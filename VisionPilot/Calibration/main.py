import os
import cv2
import json
import tqdm
import numpy as np


"""
Camera Calibration Protocol for End-to-End Models
of VisionPilot's Perception Stacks

- A quick demo
"""


def get_rotation_matrix(
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float
):
    """
    Convert Euler angles (in degrees) to a rotation matrix.

    Parameters:
        - pitch_deg: pitch (x-angle rotation), in degrees
        - yaw_deg: yaw (y-angle rotation), in degrees
        - roll_deg: roll (z-angle rotation), in degrees

    Returns:
        - R: a 3x3 rotation matrix that later can be used to
             warp the original frame to match standard pose
             perspective.
    """

    pitch, yaw, roll = np.radians([
        pitch_deg,
        yaw_deg,
        roll_deg
    ])

    rot_vector = np.array(
        [pitch, yaw, roll],
        dtype = np.float64
    )

    R, _ = cv2.Rodrigues(rot_vector)

    return R


def undistort_image(
        image: np.ndarray,
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
):
    """
    Undistort a front camera image using its specific
    intrinsics matrix and distortion coefficients.

    Parameters:
        - image: original image, fetched via OpenCV library.
        - intrinsic_matrix: a 3x3 matrix containing front camera's
                            focal lengths and principal point (should
                            be clearly defined in the camera config).
        - distortion_coeffs: a vector containing front camera distortion
                             coefficients (should also be defined in
                             the camera config too).

    Returns:
        - undistorted_image: yeah as the name suggests lol.
    """

    undistorted_image = cv2.undistort(
        image,
        intrinsic_matrix,
        distortion_coeffs
    )

    return undistorted_image