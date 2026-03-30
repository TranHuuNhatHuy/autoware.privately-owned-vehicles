/** 
* @file camera_calibration.hpp
* @brief Camera calibration utilities for VisionPilot
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace autoware_pov::vision::camera {


    /**
     * @brief Camera intrinsics data structure
     * This structure holds intrinsic parameters of a camera, including:
     *  - Intrinsic matrix (K)
     *  - Distortion coefficients
     *  - Image dimensions
     */
    struct CameraIntrinsics {
        cv::Mat K;              // 3x3 intrinsic matrix
        cv::Mat dist_coeffs;    // 1x5 distortion coefficients
        int width;              // Image width
        int height;             // Image height
    };

    /**
    * @brief Camera extrinsics data structure
    * This structure holds extrinsic parameters of a camera, including:
    *  - Rotation angles (pitch, yaw, roll) in radians
    *  - Mount height in meters
    */
    struct CameraExtrinsics {
        double pitch_rad;           // Rotation around X-axis, in radians
        double yaw_rad;             // Rotation around Y-axis, in radians
        double roll_rad;            // Rotation around Z-axis, in radians
        double mount_height_m;      // Camera mount height, in meters
    };


    