/**
 * @file camera_calibration.cpp
 * @brief Implementation of deterministic camera calibration and
 * perspective warping for E2E perception models.
 */

#include "camera_calibration.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>


namespace autoware_pov::vision::camera {

    
    CameraCalibration::CameraCalibration(
        const CameraIntrinsics& inference_intrinsics,
        const CameraExtrinsics& inference_extrinsics,
        const CameraIntrinsics& standard_intrinsics,
        const CameraExtrinsics& standard_extrinsics
    ) {
        
        // 1. Store inference intrinsics

        K_inf_ = inference_intrinsics.K.clone();
        dist_coeffs_ = inference_intrinsics.dist_coeffs.clone();
        if (K_inf_.type() != CV_64F) {
            K_inf_.convertTo(K_inf_, CV_64F);  // Ensure double precision
        }
        target_size_ = cv::Size(standard_intrinsics.width, standard_intrinsics.height); // Target resolution

        // 2. Compute rotation matrices from extrinsics

        cv::Mat R_inf = getRotationMatrix(
            inference_extrinsics.pitch_rad,
            inference_extrinsics.yaw_rad,
            inference_extrinsics.roll_rad
        );
                                        
        cv::Mat R_std = getRotationMatrix(
            standard_extrinsics.pitch_rad,
            standard_extrinsics.yaw_rad,
            standard_extrinsics.roll_rad
        );

        // 3. Compute relative rotation: R_rel = R_std * R_inf^-1
        cv::Mat R_rel = R_std * R_inf.inv();

        // 4. Simulate vertical translation via focal length rescaling
        // Instead of planar homography, scale the target's vertical focal length to
        // mathematically squeeze image and correct depth estimation for E2E models.

        cv::Mat K_std_mod = standard_intrinsics.K.clone();
        if (K_std_mod.type() != CV_64F) {
            K_std_mod.convertTo(K_std_mod, CV_64F); // Ensure double precision
        }

        double scale_factor = standard_extrinsics.mounting_height / inference_extrinsics.mounting_height;
        K_std_mod.at<double>(1, 1) *= scale_factor; // Scale f_y

        // 5. Compute master homography matrix
        // H_warp = K_std_mod * R_rel * K_inf^-1
        H_warp_ = K_std_mod * R_rel * K_inf_.inv();

    }

}