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
        pitch_deg:  float,
        yaw_deg:    float,
        roll_deg:   float
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
        image:              np.ndarray,
        intrinsic_matrix:   np.ndarray,
        distortion_coeffs:  np.ndarray
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


def get_standard_intrinsics(
        w_s:    int,
        h_s:    int,
        hfov_s: int
):
    """
    Compute intrinsics matrix of the standard pose, by simple deriving
    from the standard pose's image dimensions and its HFoV.

    Parameters:
        - w_s: width of standard pose image, in pixels.
        - h_s: height of standard pose image, in pixels.
        - hfov_s: HFoV of standard pose, in degrees.

    Returns:
        - K_s: intrinsics matrix of standard pose.

    """

    # Focal length
    f_s = (w_s / 2) / np.tan(np.radians(hfov_s) / 2)

    # Intrinsics matrix
    K_s = np.array(
        [
            f_s, 0.0, w_s / 2,
            0.0, f_s, h_s / 2,
            0.0, 0.0, 1.0
        ],
        dtype = np.float64
    )

    return K_s


def get_relative_rotation(
        R_i:    np.ndarray,
        R_s:    np.ndarray
):
    """
    Compute relative rotation matrix between the inference pose
    and the standard pose.

    Parameters:
        - R_i: rotation matrix of inference pose.
        - R_s: rotation matrix of standard pose.

    Returns:
        - R_rel: relative rotation matrix for warping inference pose
                 to match standard pose perspective.

    """

    R_rel = R_s @ np.linalg.inv(R_i)

    return R_rel