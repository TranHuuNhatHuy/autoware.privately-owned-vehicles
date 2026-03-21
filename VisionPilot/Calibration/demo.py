import os
import cv2
import json
import numpy as np

from calibration import (
    get_rotation_matrix,
    undistort_image,
    get_standard_intrinsics,
    get_relative_rotation,
    simulate_vertical_translation,
    end_to_end_calibration
)


def main():

    # Load pose data from JSON config file
    with open("./configs/camera_config.json", "r") as f:
        inference_pose = json.load(f)
    with open("./configs/standard_pose_config.json", "r") as f:
        standard_pose = json.load(f)

    # Inference frame
    # Demo example uses one from Waymo Open Dataset
    inference_frame = cv2.imread("./assets/inference_waymo.jpg")

    # Undistort
    undistorted_frame = undistort_image(
        image                   = inference_frame, 
        intrinsic_matrix        = inference_pose["intrinsic_matrix"],
        distortion_coefficients = inference_pose["distortion_coefficients"]
    )

    # Standard pose intrinsics: K_s
    K_s = get_standard_intrinsics(
        w_s     = standard_pose["img_width"],
        h_s     = standard_pose["img_height"],
        hfov_s  = standard_pose["hfov"],
    )

    # Standard pose extrinsics: R_s
    R_s = get_rotation_matrix(
        pitch_deg   = standard_pose["pitch"],
        yaw_deg     = standard_pose["yaw"],
        roll_deg    = standard_pose["roll"]
    )

    # Inference pose extrinsics: R_i
    R_i = get_rotation_matrix(
        pitch_deg   = inference_pose["pitch"],
        yaw_deg     = inference_pose["yaw"],
        roll_deg    = inference_pose["roll"]
    )

    # Relative rotation from inference pose to standard pose: R_rel
    R_rel = get_relative_rotation(
        R_i = R_i,
        R_s = R_s
    )

    # Simulate vertical translation
    K_s_mod = simulate_vertical_translation(
        intrinsics  = K_s,
        H_i         = inference_pose["camera_height"],
        H_s         = standard_pose["camera_height"]
    )

    # End-to-end calibration
    calibrated_frame = end_to_end_calibration(
        inference_image     = undistorted_frame,
        modified_intrinsics = K_s_mod,
        R_rel               = R_rel,
        K_i                 = inference_pose["intrinsic_matrix"],
        w_s                 = standard_pose["img_width"],
        h_s                 = standard_pose["img_height"]
    )

    # Show result
    cv2.imshow("Calibrated Frame", calibrated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if (__name__ == "__main__"):
    main()