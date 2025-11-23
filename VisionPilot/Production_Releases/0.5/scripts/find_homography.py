import os
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
        standard_frame_path = os.path.join(
            os.path.dirname(__file__),
            "assets/standard_frame.png"
        )
        self.standard_frame = cv2.imread(standard_frame_path)
        if (self.standard_frame is None):
            raise FileNotFoundError(f"Could not read standard frame in {standard_frame_path}")

        # Crop upper part of the image to get 2880 x 1440
        CROPPED_H = 1860 - 1440
        self.standard_frame = self.standard_frame[CROPPED_H : , :]
        print(f"Standard frame cropped shape: {self.standard_frame.shape}")
        # Save cropped standard frame for reference
        cv2.imwrite(
            os.path.join(
                os.path.dirname(__file__),
                "assets/standard_frame_cropped.png"
            ), 
            self.standard_frame
        )

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
            (159, 639),  # Left bottom
            (479, 639),  # Right bottom
            (159, 0  ),  # Left top
            (479, 0  )   # Right top
        ]

        self.bev_size = (640, 640)

        # Set homomatrix once here
        self.homography_matrix = self.compute_homography()
        print("Standard homography matrix computed:")


    def compute_homography(self):
        """
        Compute the homography matrix from the standard frame to the BEV grid.
        """

        # Convert normalized source points to pixel coordinates
        src_pts_pixel = np.array(
            [
                (
                    int(x * self.raw_w), 
                    int(y * self.raw_h)
                ) 
                for x, y in self.src_points
            ],
            dtype = np.float32
        )

        dst_pts_np = np.array(
            self.dst_points, 
            dtype = np.float32
        )

        # Compute homography matrix
        H, _ = cv2.findHomography(src_pts_pixel, dst_pts_np)

        return H


    def warp_to_bev(self, image):
        """
        Warp an image to the BEV grid using the computed homography matrix.
        """

        bev_image = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.bev_size
        )

        return bev_image


# Just for testing
if __name__ == "__main__":

    bev_homography = BEVHomography()

    # Use a random frame
    test_frame_path = ""
    test_frame_image = cv2.imread(test_frame_path)[420 : , :]
    bev_image = bev_homography.warp_to_bev(test_frame_image)

    # Save or display the BEV image as needed
    cv2.imwrite("./assets/test_bev_image.png", bev_image)