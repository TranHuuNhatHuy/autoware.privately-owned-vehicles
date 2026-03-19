# Camera Calibration Protocol for VisionPilot End-to-End Perception Stack

## I. Introduction

VisionPilot's primary end-to-end perception heads are trained
mainly on Zenesact dataset, whose camera pose and intrinsics are 
strictly defined. Such requirements will be clearly stated in the 
preliminary setup procedures for OEMs, Tier-1 suppliers and 
affiliated parties who implement our solution (hereafter referred 
to as "users").

However, even if the users correctly acquire and set up the front 
camera, differences in mounting height, position, and camera 
intrinsics are still to be expected. For example, an SUV will have 
a slightly higher mounting height than a sedan, resulting in 
different camera height and position.

Although current VisionPilot 1.0 works only in the 2D visual 
space, which means it's supposed to be unaffected by such 
differences, its end-to-end models, trained to output world 
coordinate information - road curvature, relative distance, 
relative speed, etc., could still be affected by those small 
mounting differences. 

For this, a correction mechanism must be implemented to correct 
such camera extrinsics/intrinsics differences, making it widely 
versatile and adaptable to camera specifications of various users, 
without the need of retraining those end-to-end models.

## II. Methodology and pipeline

### 1. Methodology

The goal of this camera calibration pipeline is to standardize the 
visual inputs of the VisionPilot, in a way that the perspective is 
aligned with those defined by the Zenesact dataset. In particular:

- Camera intrinsics
- Camera extrinsics (assumed the camera is placed horizontally 
centered of the ego vehicle, which is a strict requirement of 
VisionPilot) :
    - Camera height
    - Camera pitch, yaw, roll
- Horizontal Field of View (HFoV)
- Input frame dimension (1024 x 512 pixels)

### 2. Configs & parameters

The users are required to provide a configuration profile 
(hereafter referred to as "config") containing these information:

- Distortion coefficients: `k1`, `k2`, `p1`, `p2`, `p3`
- Intrinsics matrix (3x3)
- Pitch, roll and yaw angle
- Mounting height
- Input frame dimension
- HFoV

This input config is referred as inference pose, and will later, 
through this pipeline, be calibrated to those matching Zenesact's, 
referred as standard pose.

### 3. Pipeline

The pipeline follows this specific flow:

1. Fetching raw input frame from the user's camera.
2. Undistorting raw input frame to resolve pinhole effect using 
distortion coefficients.
3. Match pitch, yaw, roll of standard pose via perspective 
rotation homography calculation.
4. Height compression to simulate height difference between 
inference and standard poses.
5. Final warp from inference pose to standard pose.

### 4. Results

TBD