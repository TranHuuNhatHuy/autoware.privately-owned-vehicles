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