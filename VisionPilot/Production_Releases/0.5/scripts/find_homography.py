import cv2
import numpy as np

"""

This script finds the homography matrix to transform a cropped image of a road scene
into a bird eye view (BEV) grid. The homography is computed using manually defined source
and destination points.

Standard frame:
- Location: VisionPilot/Production_Releases/0.5/scripts/assets/standard_frame.jpg
- Resolution: 2880 x 1860
- Cropped resolution: 2880 x 1440
- BEV grid resolution: 640 x 640

This output BEV grid that can be used for lane detection and other applications.

"""


class BEVHomography:

    def __init__(self):
        """
        Initialize the BEVHomography class by loading the standard frame and defining
        source and destination points.
        """

        # Load standard frame image
        self.standard_frame = cv2.imread(
            "./assets/standard_frame.jpg"
        )

        # Crop upper part of the image to get 2880 x 1440
        CROPPED_H = 1860 - 1440
        self.standard_frame = self.standard_frame[CROPPED_H : , :]
        print(f"Standard frame cropped shape: {self.standard_frame.shape}")

        # MANUALLY DEFINED SOURCE POINTS
        # Define manually 4 source points by picking the best frame where car 
        # is perfectly straight (so egoleft/right make perfect trapezoid)
        # Four source points will be in the order: 
        # - Left bottom
        # - Right bottom
        # - Left top
        # - Right top
        # Supported by GIMP to get pixel coordinates
        RAW_SRC_POINTS = [
            (448  , 1439),  # Left bottom
            (2108 , 1439),  # Right bottom
            (1380 , 629 ),  # Left top
            (1488 , 629 )   # Right top
        ]

        # Source points (normalized)
        self.raw_h, self.raw_w = self.standard_frame.shape[:2]
        self.src_points = [
            (
                x / self.raw_w, 
                y / self.raw_h
            ) 
            for x, y in RAW_SRC_POINTS
        ]

        # Destination points, for a BEV grid of 640 x 640
        # They are NOT normalized
        self.dst_points = [
            (159, 639),  # left bottom
            (479, 639),  # right bottom
            (159, 0),    # left top
            (479, 0)     # right top
        ]

        self.bev_size = (640, 640)


    def compute_homography(self, cropped_image):
        """
        Compute the homography matrix from the cropped image to the BEV grid.
        """

        # Convert normalized source points to pixel coordinates
        src_pts_pixel = [
            (
                int(x * self.raw_w), 
                int(y * self.raw_h)
            ) 
            for x, y in self.src_points
        ]

        # Convert to numpy arrays
        src_pts_np = cv2.convertPointsToHomogeneous(np.array(src_pts_pixel)).reshape(-1, 2)
        dst_pts_np = cv2.convertPointsToHomogeneous(np.array(self.dst_points)).reshape(-1, 2)

        # Compute homography matrix
        homography_matrix, _ = cv2.findHomography(src_pts_np, dst_pts_np)

        return homography_matrix


    def warp_to_bev(self, cropped_image):
        """
        Warp the cropped image to the BEV grid using the computed homography matrix.
        """
        
        homography_matrix = self.compute_homography(cropped_image)
        bev_image = cv2.warpPerspective(
            cropped_image, 
            homography_matrix, 
            self.bev_size
        )

        return bev_image


